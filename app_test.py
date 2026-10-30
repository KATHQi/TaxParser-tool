"""
Tax Document Parser - Flask API Service
端口: 8001
功能: 接收服务器路径，处理图像，返回解析结果和可视化
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.exceptions import BadRequest, NotFound
import traceback
import shutil
import uuid
from werkzeug.utils import secure_filename
import requests
from urllib.parse import urlparse, unquote
import mimetypes
import pandas as pd
from tax_postprocessor import postprocess_internal_results


# 必须尽早设置，且在任何导入会触发 Paddle/BLAS 初始化的模块之前
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
os.environ.setdefault('KMP_AFFINITY', 'granularity=thread,compact,1,0')




# 导入原始算法模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'codeData'))
from tax_document_parser import TaxDocumentParser

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

import signal

STDIO_LOG_PATH = "/data/tensorflow/kath/Service/TaxParser/log/flask_nohup.log"

def reopen_stdio(signum, frame):
    """
    收到 SIGHUP 后，重新打开 stdout/stderr 指向同一个日志文件。
    这样配合 logrotate 的 rename 轮转，就不会出现 NUL 洞。
    """
    try:
        f = open(STDIO_LOG_PATH, "a", buffering=1)  # 行缓冲
        os.dup2(f.fileno(), 1)  # stdout
        os.dup2(f.fileno(), 2)  # stderr
        logger.info("Reopened stdout/stderr after log rotation (SIGHUP).")
    except Exception as e:
        # 这里不能依赖 logger 一定可用，所以也打印到 stderr
        try:
            print(f"[reopen_stdio] failed: {e}", file=sys.stderr)
        except Exception:
            pass

signal.signal(signal.SIGHUP, reopen_stdio)


# 创建 Flask 应用
app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# 配置
app.config['JSON_AS_ASCII'] = False
app.config['ENV'] = 'production'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB 最大请求大小

# 统一的 JSON 响应，确保不转义中文并包含 charset
def make_json_response(obj, status=200):
    return app.response_class(
        response=json.dumps(obj, ensure_ascii=False, default=str),
        status=status,
        mimetype='application/json; charset=utf-8'
    )


# 下游接口固定使用的表格字段。无论上游表格模型返回 4 列、6 列或异常列名，
# 最终每一行都按该顺序输出；无法精确匹配的字段保留为空字符串。
STANDARD_TABLE_COLUMNS = [
    '原凭证号',
    '税种',
    '品目名称',
    '税款所属时期',
    '入（退）库日期',
    '实缴（退）金额'
]


def _normalize_table_column_name(column_name):
    """
    统一表格列名中的空白和括号形式。

    例如：
    入(退)库日期 -> 入（退）库日期
    实缴(退)金额 -> 实缴（退）金额
    """
    text = ''.join(str(column_name).split())

    return text.translate(str.maketrans({
        '(': '（',
        ')': '）',
        '﹙': '（',
        '﹚': '）'
    }))


def standardize_table_row(row_dict):
    """
    将上游动态表格行转换为固定六列结构。

    仅在去除空白后的列名与标准列名完全一致时复制值；异常列、合并列和
    未知列不会进入下游结果。None/NaN 统一转换为空字符串。
    """
    normalized_standard_columns = {
        _normalize_table_column_name(column): column
        for column in STANDARD_TABLE_COLUMNS
    }
    standardized_row = {column: '' for column in STANDARD_TABLE_COLUMNS}

    if not isinstance(row_dict, dict):
        return standardized_row

    for source_column, source_value in row_dict.items():
        normalized_source_column = _normalize_table_column_name(source_column)
        target_column = normalized_standard_columns.get(normalized_source_column)
        if target_column is None:
            continue

        try:
            is_empty = source_value is None or bool(pd.isna(source_value))
        except (TypeError, ValueError):
            is_empty = source_value is None

        standardized_row[target_column] = '' if is_empty else str(source_value).strip()

    return standardized_row


def standardize_table_dataframe(dataframe): 
    """将模型返回的 DataFrame 统一转换为固定六列 DataFrame。"""
    if dataframe is None or dataframe.empty:
        return pd.DataFrame(columns=STANDARD_TABLE_COLUMNS)

    # 列名为 0、1、2... 时，表格模型通常把真实表头放在了第一行。
    has_proper_columns = not all(
        isinstance(column, int) or str(column).isdigit()
        for column in dataframe.columns
    )

    if has_proper_columns:
        column_names = [
            _normalize_table_column_name(column)
            for column in dataframe.columns
        ]
        data_start_row = 0
    else:
        from difflib import SequenceMatcher

        column_names = [
            _normalize_table_column_name(value)
            for value in dataframe.iloc[0].tolist()
        ]
        template_4col = ['税种', '税款所属时期', '入（退）库日期', '实缴（退）金额']

        column_text = ''.join(column_names)
        similarity_6col = SequenceMatcher(
            None, column_text, ''.join(STANDARD_TABLE_COLUMNS)
        ).ratio()
        similarity_4col = SequenceMatcher(
            None, column_text, ''.join(template_4col)
        ).ratio()
        max_similarity = max(similarity_6col, similarity_4col)

        if max_similarity >= 0.8:
            # 第一行高度接近标准表头，因此只把后续行作为数据。
            data_start_row = 1
        else:
            # 无法可靠识别表头时，仅按已知的 4/6 列模板建立列名。
            if len(column_names) == 4:
                column_names = template_4col
            elif len(column_names) == 6:
                column_names = STANDARD_TABLE_COLUMNS
            data_start_row = 0

    standardized_rows = []
    for row_index in range(data_start_row, len(dataframe)):
        source_row = dataframe.iloc[row_index]
        source_row_dict = {
            column_name: source_row.iloc[column_index]
            for column_index, column_name in enumerate(column_names)
            if column_index < len(source_row)
        }
        standardized_rows.append(standardize_table_row(source_row_dict))

    return pd.DataFrame(standardized_rows, columns=STANDARD_TABLE_COLUMNS)


# 全局变量
parser_instance = None
results_cache = {}
import threading
# 限制 /api/parse 并发数（进程内限制，若使用 gunicorn 多 worker，请使用分布式锁）
parse_semaphore = threading.BoundedSemaphore(value=5)


# def get_parser():
#     """获取或创建解析器实例"""
#     global parser_instance
#     if parser_instance is None:
#         logger.info("初始化 TaxDocumentParser...")
#         parser_instance = TaxDocumentParser(
#             output_dir=os.path.join(os.path.dirname(__file__), 'results'),
#             use_gpu=True
#         )
#     return parser_instance
def get_parser():
    """获取或创建解析器实例（在 worker 进程中延迟初始化）"""
    global parser_instance
    if parser_instance is None:
        logger.info("初始化 TaxDocumentParser...")
        # 局部导入，确保环境变量已设置且在 worker 进程中初始化模型
        from tax_document_parser import TaxDocumentParser
        parser_instance = TaxDocumentParser(
            output_dir=os.path.join(os.path.dirname(__file__), 'results'),
            use_gpu=True
        )
    return parser_instance


def enhance_table_html(html_string, dataframe):
    """
    增强表格 HTML：添加列标题、格式化等
    
    Args:
        html_string: 原始 HTML 字符串
        dataframe: pandas DataFrame
        
    Returns:
        增强后的 HTML 字符串
    """
    if not html_string or dataframe is None or dataframe.empty:
        return html_string
    
    try:
        import re
        from bs4 import BeautifulSoup
        
        # 使用 BeautifulSoup 解析 HTML
        soup = BeautifulSoup(html_string, 'html.parser')
        table = soup.find('table')
        
        if not table:
            return html_string
        
        # 检查是否已有 thead
        thead = table.find('thead')
        tbody = table.find('tbody')
        
        if not thead and tbody:
            # 创建 thead 和第一行作为列标题
            thead = soup.new_tag('thead')
            tr_header = soup.new_tag('tr')
            
            # 从 dataframe 列数推断列标题
            num_cols = len(dataframe.columns) if hasattr(dataframe, 'columns') else dataframe.shape[1]
            for col_idx in range(num_cols):
                th = soup.new_tag('th')
                th.string = f'列 {col_idx + 1}'
                tr_header.append(th)
            
            thead.append(tr_header)
            
            # 将 thead 插入到 table 中
            if tbody:
                tbody.insert_before(thead)
            else:
                table.append(thead)
        
        # 添加表格属性
        table['style'] = 'width: 100%; border-collapse: collapse; background: white;'
        
        # 为所有 th 和 td 添加样式
        for th in table.find_all('th'):
            th['style'] = 'border: 1px solid #e0e0e0; padding: 12px; text-align: left; background: #f8f9fa; font-weight: 600; color: #333; font-size: 13px;'
        
        for td in table.find_all('td'):
            td['style'] = 'border: 1px solid #e0e0e0; padding: 12px; text-align: left; font-size: 13px;'
        
        return str(table)
    
    except Exception as e:
        logger.warning(f"增强表格 HTML 失败: {e}，返回原始 HTML")
        return html_string


def format_results_for_frontend(results, directory_path):
    """格式化处理结果用于前端展示"""
    formatted_results = {
        'status': 'success',
        'directory': directory_path,
        'timestamp': datetime.now().isoformat(),
        'total_images': len(results),
        'successful': sum(1 for r in results if r.get('status') == 'success'),
        'failed': sum(1 for r in results if r.get('status') == 'error'),
        'no_seal': sum(1 for r in results if r.get('status') == 'no-seal'),
        'wrong_seal': sum(1 for r in results if r.get('status') == 'wrong-seal'),
        'images': []
    }
    
    for i, result in enumerate(results):
        if result.get('status') == 'error':
            formatted_results['images'].append({
                'index': i + 1,
                'status': 'error',
                'message': result.get('message', '未知错误'),
                'image_path': result.get('image_path', '')
            })
        elif result.get('status') == 'wrong-seal':
            logger.info(f"DEBUG: wrong-seal 状态, result keys = {list(result.keys())}")
            # logger.info(f"DEBUG: extracted_fields = {result.get('extracted_fields')}")
            formatted_results['images'].append({
                'index': i + 1,
                'status': 'wrong-seal',
                'message': '公章错误',
                'image_path': result.get('image_path', ''),
                'extracted_fields': result.get('extracted_fields', {})
            })
        elif result.get('status') == 'no-seal':
            logger.info(f"DEBUG: no-seal 状态, result keys = {list(result.keys())}")
            # logger.info(f"DEBUG: extracted_fields = {result.get('extracted_fields')}")
            formatted_results['images'].append({
                'index': i + 1,
                'status': 'no-seal',
                'message': '无公章',
                'image_path': result.get('image_path', ''),
                'extracted_fields': result.get('extracted_fields', {})
            })
        else:
            # 格式化成功结果
            image_data = {
                'index': i + 1,
                'status': 'success',
                'image_path': result.get('image_path', ''),
                'labels': result.get('labels', []),
                'grouped_data': {},
                'tables': {},
                'extracted_fields': result.get('extracted_fields', {})  # 直接使用后端提取的字段
            }
            
            # TAX_TABLE_REPORT_FRONTEND_V1 (opt-in; default response schema unchanged)
            if os.environ.get('TAX_POSTPROCESS_REPORT', '0') == '1':
                image_data['postprocess'] = result.get('_postprocess_report', {})

            # 处理分组数据 - 保留完整的 OCR 检测结果
            grouped = result.get('grouped_data', {})
            for label, items in grouped.items():
                image_data['grouped_data'][label] = {
                    'count': len(items),
                    'items': items,  # 保留所有项目用于展开
                    'total': len(items)
                }
            
            # 处理表格结果 - 保留 HTML 用于直接展示
            table_results = result.get('table_results', {})
            # logger.info(f"\n========== 表格数据处理 ==========")
            # logger.info(f"图像 #{i+1}: {result.get('image_path', 'unknown')}")
            # logger.info(f"table_results 类型: {type(table_results)}")
            # logger.info(f"table_results 内容: {table_results}")
            # logger.info(f"table_results 键: {list(table_results.keys())}")
            # logger.info(f"table_results 是否为空: {len(table_results) == 0}")
            
            for table_type, table_data in table_results.items():
                # logger.info(f"\n[{table_type}]:")
                # logger.info(f"  table_data 类型: {type(table_data)}")
                # logger.info(f"  table_data 是否为 None: {table_data is None}")
                
                # 前端展示也必须使用固定六列，不能继续返回模型生成的原始 HTML。
                if table_data and isinstance(table_data, dict):
                    standardized_df = standardize_table_dataframe(
                        table_data.get('dataframe')
                    )
                    standardized_html = standardized_df.to_html(index=False, border=0)
                    enhanced_html = enhance_table_html(
                        standardized_html, standardized_df
                    )

                    image_data['tables'][table_type] = {
                        'html': enhanced_html,
                        'rows': standardized_df.shape[0],
                        'cols': standardized_df.shape[1]
                    }
                    logger.info(
                        f"  ✓ {table_type} 已按固定六列添加到前端返回数据"
                    )
                else:
                    logger.info(f"  ✗ table_data 为空或不存在")
                    # 即使 table_data 为 None，也要添加占位符，这样前端能知道这个表格存在
                    image_data['tables'][table_type] = {
                        'html': '',
                        'rows': 0,
                        'cols': 0,
                        'status': 'empty'
                    }
                    logger.info(f"  ℹ {table_type} 已添加空表格占位符 (None值)")
            
            formatted_results['images'].append(image_data)
    
    return formatted_results


# ==================== API 路由 ====================

@app.route('/', methods=['GET'])
def index():
    """主页面"""
    return render_template('index.html')


def download_file(url, save_dir):
    """从 URL 下载文件到指定目录"""
    try:
        logger.info(f"正在下载: {url}")
        # verify=False 忽略 SSL 证书验证，避免内部系统证书问题
        response = requests.get(url, stream=True, verify=False, timeout=60)
        response.raise_for_status()
        
        # 尝试从 Content-Disposition 获取文件名
        filename = None
        content_disposition = response.headers.get('Content-Disposition')
        if content_disposition:
            import re
            fname = re.findall('filename="?([^"]+)"?', content_disposition)
            if fname:
                filename = fname[0]
        
        # 如果没有，从 URL 获取
        if not filename:
            parsed_url = urlparse(url)
            path = unquote(parsed_url.path)
            filename = os.path.basename(path)
            
        # 如果还是没有，生成随机文件名
        if not filename or filename == '.' or filename == '/':
            ext = mimetypes.guess_extension(response.headers.get('content-type', '')) or '.pdf'
            filename = f"{uuid.uuid4()}{ext}"
            
        # 简单处理非法字符
        filename = filename.replace('/', '_').replace('\\', '_').replace(':', '_')
        
        save_path = os.path.join(save_dir, filename)
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        logger.info(f"下载完成: {save_path}")
        return save_path
    except Exception as e:
        logger.error(f"下载失败 {url}: {str(e)}")
        raise


def _can_use_text_pdf_extract(pdf_path):
    """判断 PDF 是否包含可直接提取的文本层。"""
    try:
        import fitz
        total_text_len = 0
        with fitz.open(pdf_path) as doc:
            for page in doc:
                total_text_len += len((page.get_text("text") or "").strip())
                if total_text_len >= 30:
                    return True
        return False
    except Exception as e:
        logger.warning(f"检查 PDF 文本层失败，转 OCR: {pdf_path}, err={e}")
        return False


def _build_internal_result_from_extractor(pdf_path, extract_result):
    """将 tax_certificate_extractor 输出适配为现有接口内部 result 结构。"""
    details_df = extract_result.get('details_df')
    if details_df is None:
        details_df = pd.DataFrame(extract_result.get('details', []))
    if details_df is None:
        details_df = pd.DataFrame()
    details_df = details_df.fillna('')

    total_cn = extract_result.get('total_amount_cn', '') or ''
    total_raw = extract_result.get('total_amount_raw', '') or ''

    table_html = details_df.to_html(index=False, border=0) if not details_df.empty else ''
    table_results = {
        'table1': {
            'dataframe': details_df,
            'html': table_html
        }
    }

    labels = []
    if not details_df.empty and '税种' in details_df.columns:
        labels = [str(x).strip() for x in details_df['税种'].tolist() if str(x).strip()]
        labels = list(dict.fromkeys(labels))
    # print('\n\naaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  ',extract_result.get('fill_person', '') or '')
    return {
        'status': 'success',
        'image_path': str(pdf_path),
        '_pdf_source': str(pdf_path),
        '_page_index': extract_result.get('page_index', 1),
        '_total_pages': extract_result.get('total_pages', 1),
        'labels': labels,
        'grouped_data': {},
        'table_results': table_results,
        'extracted_fields': {
            '编号': extract_result.get('certificate_no', '') or '',
            '税务机关': extract_result.get('tax_authority', '') or '',
            '填发日期': extract_result.get('issue_date', '') or '',
            '纳税人名称': extract_result.get('taxpayer_name', '') or '',
            '纳税人识别号': extract_result.get('taxpayer_id', '') or '',
            '填票人': extract_result.get('fill_person', '') or '',
            '金额合计': [total_cn, total_raw]
        }
    }


def _try_extract_pdf_without_ocr(pdf_path):
    """尝试直接读取 PDF 文本提取；失败返回 None 供上层回退 OCR。
    成功时返回每页结果的列表。"""
    if not _can_use_text_pdf_extract(pdf_path):
        return None

    try:
        from tax_certificate_extractor import extract_pdf_by_pages
        page_results = extract_pdf_by_pages(pdf_path)
        if not page_results:
            return None
        return [_build_internal_result_from_extractor(pdf_path, r) for r in page_results]
    except Exception as e:
        logger.warning(f"PDF 直读提取失败，回退 OCR: {pdf_path}, err={e}")
        return None


def _process_directory_logic(directory_path, recursive, request_tag):
    """
    核心处理逻辑：处理指定目录下的文件
    """
    acquired = False
    temp_dir = None
    try:
        # 限制并发
        acquired = parse_semaphore.acquire(blocking=False)
        if not acquired:
            return make_json_response({
                'status': 'error',
                'message': '服务器繁忙，请稍后重试'
            }, 429)
            
        # 验证路径
        if not os.path.isdir(directory_path):
            return make_json_response({
                'status': 'error',
                'message': f'目录不存在: {directory_path}'
            }, 404)
        
        logger.info(f"开始处理目录: {directory_path} (recursive={recursive}, requesttag={request_tag})")
        
        # 扫描目录，找到所有文件（包括 PDF 和图像）
        from pathlib import Path
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        pdf_extension = '.pdf'
        
        all_files = []
        if recursive:
            all_files = [str(p) for p in Path(directory_path).rglob('*')
                        if p.suffix.lower() in image_extensions or p.suffix.lower() == pdf_extension]
        else:
            all_files = [str(p) for p in Path(directory_path).glob('*')
                        if p.suffix.lower() in image_extensions or p.suffix.lower() == pdf_extension]
        
        logger.info(f"找到 {len(all_files)} 个文件")
        
        # 分离 PDF 和图像文件
        pdf_files = [f for f in all_files if f.lower().endswith('.pdf')]
        image_files = [f for f in all_files if not f.lower().endswith('.pdf')]
        
        logger.info(f"  - PDF 文件: {len(pdf_files)} 个")
        logger.info(f"  - 图像文件: {len(image_files)} 个")

        results = []

        # 优先尝试 PDF 文本层直提，失败再进入 OCR
        ocr_pdf_files = []
        for pdf_path in pdf_files:
            direct_results = _try_extract_pdf_without_ocr(pdf_path)
            if direct_results is not None:
                logger.info(f"PDF 直提成功: {pdf_path} ({len(direct_results)} 页)")
                results.extend(direct_results)
            else:
                ocr_pdf_files.append(pdf_path)

        logger.info(f"PDF 直提成功: {len(pdf_files) - len(ocr_pdf_files)} 个, 回退 OCR: {len(ocr_pdf_files)} 个")
        
        # 处理 PDF 文件：先转换为图像
        pdf_to_images_map = {}  # {pdf_path: [(page_idx, image_obj), ...]}
        if ocr_pdf_files:
            # 使用 UUID 创建唯一的临时目录，避免并发冲突
            temp_dir = os.path.join(os.path.dirname(__file__), 'temp_pdf_images', str(uuid.uuid4()))
            os.makedirs(temp_dir, exist_ok=True)

            for pdf_path in ocr_pdf_files:
                try:
                    from pdf2image import convert_from_path
                    logger.info(f"转换 PDF(OCR): {pdf_path}")
                    images = convert_from_path(pdf_path, dpi=300)
                    logger.info(f"  页数: {len(images)}")

                    # 保存每页为临时图像文件，并让 parser 处理
                    page_images = []
                    for page_idx, img in enumerate(images, 1):
                        temp_image_path = os.path.join(temp_dir, f"PDF_{Path(pdf_path).stem}_page{page_idx}.jpg")
                        img.save(temp_image_path, 'JPEG')
                        page_images.append((page_idx, temp_image_path))

                    pdf_to_images_map[pdf_path] = page_images
                    image_files.extend([img_path for _, img_path in page_images])

                except ImportError:
                    logger.error(f"pdf2image 库未安装，跳过 PDF: {pdf_path}")
                except Exception as e:
                    logger.error(f"PDF 转换失败 {pdf_path}: {str(e)}")
        
        # 去重：避免重复处理同一文件
        image_files = list(set(image_files))
        logger.info(f"去重后待处理图像数量: {len(image_files)}")
        
        # 处理所有图像文件（包括从 PDF 转换的）
        if image_files:
            parser = get_parser()
            for img_path in image_files:
                try:
                    logger.info(f"处理: {img_path}")
                    result = parser.process_image(img_path)
                    # print('【app 返回内容】',result )
                    if result['status'] =='wrong-seal':
                        result['message'] = '公章错误'
                        results.append(result)
                        continue
                    if result['status'] =='no-seal':
                        result['message'] = '无公章'
                        results.append(result)
                        continue
                    results.append(result)
                except Exception as e:
                    import traceback
                    logger.error(f"处理图像失败 {img_path}: {str(e)}\n{traceback.format_exc()}")
                    results.append({
                        'status': 'error',
                        'image_path': img_path,
                        'message': str(e)
                    })
        
        # print('\n\n\n\nresults:',len(results),'个结果:'+'\n\n\n') #+str(results)
        
        # TAX_TABLE_POSTPROCESS_V1
        # Single hook for OCR + text PDFs; runs before frontend/API formatting.
        if os.environ.get('TAX_POSTPROCESS_ENABLED', '1') != '0':
            try:
                results = postprocess_internal_results(
                    results,
                    thousands=os.environ.get('TAX_POSTPROCESS_THOUSANDS', '1') != '0'
                )
            except Exception:
                # Keep original recognition output if optional repair fails.
                logger.exception('Tax table post-processing failed; preserving original results')

        # 检查是否为前端请求 (requesttag=1)
        if request_tag == 1:
            # ============ 返回前端专有格式 ============
            formatted = format_results_for_frontend(results, directory_path)
            
            # 缓存结果
            cache_key = datetime.now().isoformat()
            results_cache[cache_key] = results
            formatted['cache_key'] = cache_key
            # print('   【DEBUG formatted】', formatted)
            logger.info(f"处理完成: 成功 {formatted['successful']}, 失败 {formatted['failed']}, 无公章 {formatted['no_seal']}, 公章错误 {formatted['wrong_seal']}")
            
            # 调试：打印返回给前端的数据中的表格信息
            # logger.info("\n========== 返回给前端的表格数据 ==========")
            # for img_data in formatted['images']:
            #     logger.info(f"图像 #{img_data['index']}: 表格数量 = {len(img_data['tables'])}")
            #     for table_type, table_info in img_data['tables'].items():
            #         logger.info(f"  - {table_type}: {table_info}")
            # logger.info("=" * 50)
            
            return make_json_response({
                'status': 'success',
                'data': formatted
            }, 200)
        else:
            # ============ 返回 mock API 兼容格式 ============
            # 首先聚合结果：将来自同一个 PDF 的多页结果合并
            file_results_map = {}  # {original_file: [results]}
            
            for result in results:
                image_path = result.get('image_path', 'unknown')
                
                # 检查是否是从 PDF 转换的临时图像
                is_temp_pdf_image = bool(temp_dir) and (temp_dir in image_path)

                # 检查是否是直提 PDF 结果（有 _pdf_source 标记）
                pdf_source = result.get('_pdf_source')

                if pdf_source:
                    # 直提 PDF 结果：按原始 PDF 路径聚合，保留逐页索引
                    page_idx = result.get('_page_index', 1)
                    if pdf_source not in file_results_map:
                        file_results_map[pdf_source] = []
                    file_results_map[pdf_source].append((page_idx, result))
                elif is_temp_pdf_image:
                    # 找到对应的原始 PDF
                    original_pdf = None
                    page_idx = None
                    for pdf_path, page_list in pdf_to_images_map.items():
                        for p_idx, temp_path in page_list:
                            if temp_path == image_path:
                                original_pdf = pdf_path
                                page_idx = p_idx
                                break
                        if original_pdf:
                            break

                    if original_pdf:
                        if original_pdf not in file_results_map:
                            file_results_map[original_pdf] = []
                        file_results_map[original_pdf].append((page_idx, result))
                else:
                    # 普通图像文件
                    if image_path not in file_results_map:
                        file_results_map[image_path] = []
                    file_results_map[image_path].append((1, result))
            
            # 构建 data_list
            data_list = []
            for file_path, page_results in file_results_map.items():
                filename = os.path.basename(file_path)
                is_pdf = file_path.lower().endswith('.pdf')
                
                # 按页码排序
                page_results.sort(key=lambda x: x[0])
                
                result_items = []
                total_pages = len(page_results)
                
                for page_idx, result in page_results:
                    logger.info(f"DEBUG: 处理文件 {filename}, page_idx={page_idx}, is_pdf={is_pdf}")
                    logger.info(f"DEBUG: result status = {result.get('status')}")
                    logger.info(f"DEBUG: result keys = {list(result.keys())}")
                    
                    # 提前提取 extracted_fields，所有分支都可能用到
                    extracted_fields = result.get('extracted_fields', {})
                    
                    # 计算页码信息（所有分支都可能用到）
                    page_info = extracted_fields.get('页码') #or {'currentPage': -1, 'totalPage': -1}
                    if is_pdf:
                        page_info = {
                            'currentPage': int(page_idx),
                            'totalPage': int(total_pages)
                        }
                    
                    if result.get('status') == 'error':
                        result_items.append({
                            'status': {
                                'code': 500,
                                'message': result.get('message', '处理失败')
                            }
                            
                        })
                    elif result.get('status') == 'wrong-seal':
                        result_items.append({
                            'status': {
                                'code': 501,
                                'message': '公章错误'
                            },
                            'content': {
                                '编号': extracted_fields.get('编号', ''),
                                '页码': page_info,
                                '税务机关': extracted_fields.get('税务机关', ''),
                                '填发日期': extracted_fields.get('填发日期', ''),
                                '纳税人名称': extracted_fields.get('纳税人名称', ''),
                                '纳税人识别号': extracted_fields.get('纳税人识别号', ''),
                                '事项': '',
                                '填票人': extracted_fields.get('填票人', ''),
                                '金额合计': extracted_fields.get('金额合计', ['', ''])
                            },
                            
                        })
                    elif result.get('status') == 'no-seal':
                        result_items.append({
                            'status': {
                                'code': 502,
                                'message': '无公章'
                            },
                            'content': {
                                '编号': extracted_fields.get('编号', ''),
                                '页码': page_info,
                                '税务机关': extracted_fields.get('税务机关', ''),
                                '填发日期': extracted_fields.get('填发日期', ''),
                                '纳税人名称': extracted_fields.get('纳税人名称', ''),
                                '纳税人识别号': extracted_fields.get('纳税人识别号', ''),
                                '事项': '',
                                '填票人': extracted_fields.get('填票人', ''),
                                '金额合计': extracted_fields.get('金额合计', ['', ''])
                            },
                        })
                    
                    else:
                        # 成功情况：将解析结果转换为 mock 格式
                        result_item = {
                            'title': '中华人民共和国税收完税证明',
                            'content': {
                                '编号': extracted_fields.get('编号', ''),
                                '页码': page_info,
                                '税务机关': extracted_fields.get('税务机关', ''),
                                '填发日期': extracted_fields.get('填发日期', ''),
                                '纳税人名称': extracted_fields.get('纳税人名称', ''),
                                '纳税人识别号': extracted_fields.get('纳税人识别号', ''),
                                '事项': '',
                                '填票人': extracted_fields.get('填票人', ''),
                                '金额合计': extracted_fields.get('金额合计', ['', ''])
                            },
                            'table': [],
                            'others': [],
                            'status': {
                                'code': 200,
                                'message': 'successful'
                            }
                        }
                        
                        # TAX_TABLE_REPORT_API_V1 (opt-in)
                        if os.environ.get('TAX_POSTPROCESS_REPORT', '0') == '1':
                            result_item['postprocess'] = result.get('_postprocess_report', {})

                        # 添加识别的标签作为事项（排除 table、title 等技术标签）
                        if result.get('labels'):
                            labels = [l for l in result.get('labels', []) if l not in ['table', 'table1', 'table2', 'title', 'ink_F', 'ink_T']]
                            if labels:
                                result_item['content']['事项'] = ', '.join(labels)
                        
                        # 添加表格数据 - 自动检测列名格式
                        table_results = result.get('table_results', {})
                        logger.info(f"DEBUG: 处理 {filename} 的表格数据")
                        logger.info(f"DEBUG: table_results keys = {list(table_results.keys())}")
                        logger.info(f"DEBUG: table_results 完整内容 = {table_results}")
                        for table_type, table_data in table_results.items():
                            logger.info(f"DEBUG: 处理 table_type={table_type}")
                            # logger.info(f"DEBUG: table_data type = {type(table_data)}")
                            # logger.info(f"DEBUG: table_data value = {table_data}")
                            logger.info(f"DEBUG: isinstance(table_data, dict) = {isinstance(table_data, dict)}")
                            if table_data and isinstance(table_data, dict) and table_data.get('dataframe') is not None:
                                logger.info(f"DEBUG: table_type={table_type} 有 dataframe")
                                df = table_data.get('dataframe')
                                logger.info(f"DEBUG: df shape = {df.shape if df is not None else 'None'}")
                                logger.info(f"DEBUG: df columns = {list(df.columns) if df is not None else 'None'}")
                                if df is not None and not df.empty:
                                    standardized_df = standardize_table_dataframe(df)
                                    standardized_rows = standardized_df.to_dict('records')
                                    result_item['table'].extend(standardized_rows)
                                    logger.info(
                                        f"DEBUG: 添加 {len(standardized_rows)} 行标准化表格数据: "
                                        f"{standardized_rows}"
                                    )
                        
                        # 添加额外字段（如果是 PDF）
                        if is_pdf:
                            result_item['page'] = page_idx
                            result_item['total_pages'] = total_pages
                        
                        # 将这个页面的 result_item 添加到列表
                        result_items.append(result_item)
                
                # 将文件的所有页面结果添加到 data_list
                data_list.append({
                    'filename': filename,
                    'results': result_items
                })
            
            # 在此处尝试清理临时文件（若存在），同时 finally 中也会做一次保证性清理
            try:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    logger.info(f"已清理临时目录: {temp_dir}")
            except Exception as e:
                logger.warning(f"清理临时目录失败: {e}")
            
            # 返回 mock API 兼容格式
            response_data = {
                'code': 200,
                'message': 'successful',
                'data': data_list
            }
            
            logger.info(f"处理完成 (mock 格式): 返回 {len(data_list)} 个文件")
            
            return make_json_response(response_data, 200)
    
    except Exception as e:
        tb = traceback.format_exc()
        logger.error(f"处理失败: {str(e)}\n{tb}")
        return make_json_response({
            'status': 'error',
            'message': f'处理失败: {str(e)}\n{tb}'
        }, 500)
    finally:
        if acquired:
            try:
                parse_semaphore.release()
            except Exception as e:
                logger.warning(f"释放 parse_semaphore 失败: {e}")
        # 最终保证性清理 temp_dir，防止磁盘被占满
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"已在 finally 清理临时目录: {temp_dir}")
        except Exception as e:
            logger.warning(f"finally 中清理临时目录失败: {e}")


@app.route('/infoextract', methods=['POST'])
def parse_document():
    logger.info(f"=========================================infoextract=============================================")
    data = request.get_json()
    
    if not data:
         return make_json_response({
            'status': 'error',
            'message': '请求体为空'
        }, 400)

    # 支持 directory_path (本地路径) 或 file_url/file_urls (远程地址)
    directory_path = data.get('directory_path')
    file_url = data.get('file_url')
    file_urls = data.get('file_urls')
    
    # 兼容处理：如果传入了 file_url，转换为列表
    if file_url and not file_urls:
        file_urls = [file_url]
    
    if not directory_path and not file_urls:
        return make_json_response({
            'status': 'error',
            'message': '缺少必需参数: directory_path 或 file_url/file_urls'
        }, 400)
    
    recursive = data.get('recursive', False)
    request_tag = data.get('requesttag')
    
    temp_download_dir = None
    
    try:
        # 如果提供了 URL，先下载文件
        if file_urls:
            # 创建临时目录
            temp_download_dir = os.path.join(os.path.dirname(__file__), 'temp_downloads', str(uuid.uuid4()))
            os.makedirs(temp_download_dir, exist_ok=True)
            
            downloaded_count = 0
            for url in file_urls:
                try:
                    download_file(url, temp_download_dir)
                    downloaded_count += 1
                except Exception as e:
                    logger.warning(f"文件下载失败 {url}: {e}")
            
            if downloaded_count == 0:
                 return make_json_response({
                    'status': 'error',
                    'message': '所有文件下载失败'
                }, 400)
                
            # 使用临时目录作为处理目录
            directory_path = temp_download_dir
            recursive = False 
            
        return _process_directory_logic(directory_path, recursive, request_tag)
        
    finally:
        # 如果是临时下载目录，处理完后清理
        if temp_download_dir and os.path.exists(temp_download_dir):
            try:
                shutil.rmtree(temp_download_dir)
                logger.info(f"已清理临时下载目录: {temp_download_dir}")
            except Exception as e:
                logger.warning(f"清理临时下载目录失败: {e}")


@app.route('/upload_and_extract', methods=['POST'])
def upload_and_extract():
    logger.info(f"=========================================upload_and_extract=============================================")
    
    # 检查是否有文件上传
    if 'files' not in request.files:
        return make_json_response({'status': 'error', 'message': '没有上传文件'}, 400)
    
    files = request.files.getlist('files')
    if not files or files[0].filename == '':
        return make_json_response({'status': 'error', 'message': '未选择文件'}, 400)
        
    # 获取 requesttag 参数
    raw_tag = request.form.get('requesttag')
    try:
        request_tag = int(raw_tag) if raw_tag else None
    except ValueError:
        request_tag = None
        
    # 创建临时目录
    temp_base_dir = os.path.join(os.path.dirname(__file__), 'tempimg')
    session_id = str(uuid.uuid4())
    upload_dir = os.path.join(temp_base_dir, session_id)
    os.makedirs(upload_dir, exist_ok=True)
    
    try:
        saved_files_count = 0
        for file in files:
            if file and file.filename:
                # secure_filename 会过滤掉中文字符，改用 os.path.basename 保留原始文件名
                # filename = secure_filename(file.filename)
                filename = os.path.basename(file.filename)
                
                # 如果文件名为空，生成一个随机名
                if not filename:
                    ext = os.path.splitext(file.filename)[1]
                    filename = f"{uuid.uuid4()}{ext}"
                
                file_path = os.path.join(upload_dir, filename)
                file.save(file_path)
                saved_files_count += 1
        
        logger.info(f"已保存 {saved_files_count} 个文件到 {upload_dir}")
        
        # 调用核心处理逻辑
        # 将获取到的 request_tag 传入
        response = _process_directory_logic(upload_dir, recursive=False, request_tag=request_tag)
        
        return response
        
    except Exception as e:
        logger.error(f"文件上传处理失败: {e}")
        return make_json_response({'status': 'error', 'message': str(e)}, 500)
    finally:
        # 清理上传的临时目录
        try:
            if os.path.exists(upload_dir):
                shutil.rmtree(upload_dir)
                logger.info(f"已清理上传目录: {upload_dir}")
        except Exception as e:
            logger.warning(f"清理上传目录失败: {e}")


@app.route('/api/results/<cache_key>', methods=['GET'])
def get_cached_results(cache_key):
    """获取缓存的详细结果"""
    try:
        if cache_key not in results_cache:
            return make_json_response({
                'status': 'error',
                'message': '缓存结果不存在或已过期'
            }, 404)
        
        results = results_cache[cache_key]
        return make_json_response({
            'status': 'success',
            'data': results
        }, 200)
    
    except Exception as e:
        logger.error(f"获取缓存失败: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'获取缓存失败: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查"""
    return make_json_response({
        'status': 'ok',
        'service': 'Tax Document Parser API',
        'version': '1.0.0',
        'timestamp': datetime.now().isoformat()
    }, 200)


@app.route('/api/status', methods=['GET'])
def status():
    """获取服务状态"""
    return make_json_response({
        'status': 'running',
        'service': 'Tax Document Parser API',
        'port': 8001,
        'cache_size': len(results_cache),
        'timestamp': datetime.now().isoformat()
    }, 200)


# ==================== 错误处理 ====================

@app.errorhandler(BadRequest)
def bad_request(error):
    return make_json_response({'status': 'error', 'message': '请求格式错误'}, 400)


@app.errorhandler(NotFound)
def not_found(error):
    return make_json_response({'status': 'error', 'message': '资源不存在'}, 404)


@app.errorhandler(500)
def internal_error(error):
    logger.error(f"内部错误: {str(error)}")
    return make_json_response({'status': 'error', 'message': '服务器内部错误'}, 500)




# ==================== 主程序 ====================

if __name__ == '__main__':
    logger.info("="*60)
    logger.info("Tax Document Parser - Flask API Service")
    logger.info("="*60)
    
    # 检查是否存在 SSL 证书
    ssl_context = None
    if os.path.exists('cert.pem') and os.path.exists('key.pem'):
        ssl_context = ('cert.pem', 'key.pem')
        logger.info("启用 HTTPS (使用 cert.pem/key.pem)")
        protocol = "https"
    else:
        logger.info("未找到证书文件，使用 HTTP 模式 (若需 HTTPS 请提供 cert.pem 和 key.pem)")
        protocol = "http"
        
    logger.info(f"服务器启动于 {protocol}://0.0.0.0:8889")
    logger.info(f"文档主页: {protocol}://localhost:8889")
    logger.info("="*60)
    
    app.run(
        host='0.0.0.0',
        port=8889,
        threaded=True,
        ssl_context=ssl_context
    )
 