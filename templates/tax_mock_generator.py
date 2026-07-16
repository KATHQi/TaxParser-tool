import random
import datetime
import string
import uuid


def random_date(start_year=1990, end_year=2025):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    delta = end - start
    return (start + datetime.timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")


def random_chinese_name(length=3):
    chars = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊於惠甄魏加封任鲁顾孟"
    return "".join(random.choice(chars) for _ in range(length))


def random_money(min_value=100.0, max_value=100000.0):
    amount = round(random.uniform(min_value, max_value), 2)
    return f"¥{amount:,.2f}"


def generate_bianhao():
    """Generate either 'No. '+18digits or 'XX（YYYY）证明 '+9digits, where XX（YYYY） can vary."""
    if random.random() < 0.5:
        # No. +18 digits
        return "No. " + ''.join(random.choices(string.digits, k=18))
    else:
        prefix_num = random.randint(10, 99)
        inner = ''.join(random.choices(string.digits, k=4))
        prefix = f"{prefix_num}（{inner}）证明 "
        suffix = ''.join(random.choices(string.digits, k=9))
        return prefix + suffix


def tex_like_table_rows():
    rows = [
        ("增值税(滞纳金)", "2023-05-01 至 2023-05-31", "2024-03-05", "¥3097.85"),
        ("增值税", "2023-05-01 至 2023-05-31", "2024-03-05", "¥23468.54"),
        ("企业所得税", "2023-01-01 至 2023-12-31", "2024-05-09", "¥20424.63"),
        ("企业所得税", "2024-01-01 至 2024-03-31", "2024-04-12", "¥9208.57"),
        ("城市维护建设税", "2023-05-01 至 2023-05-31", "2024-03-05", "¥821.40"),
        ("城市维护建设税(滞纳金)", "2023-05-01 至 2023-05-31", "2024-03-05", "¥108.42"),
        ("印花税", "2023-10-01 至 2023-12-31", "2024-01-10", "¥9071.31"),
        ("印花税", "2024-01-01 至 2024-03-31", "2024-04-10", "¥8488.76"),
        ("印花税", "2024-04-01 至 2024-06-30", "2024-07-10", "¥9280.19"),
        ("印花税", "2024-07-01 至 2024-09-30", "2024-10-09", "¥6375.57"),
        ("教育费附加", "2023-05-01 至 2023-05-31", "2024-03-05", "¥352.03"),
        ("地方教育附加", "2023-05-01 至 2023-05-31", "2024-03-05", "¥234.68"),
    ]
    k = random.randint(1, len(rows))
    chosen = random.sample(rows, k)
    out = []
    for t, period, date, amt in chosen:
        out.append({
            "税种": t,
            "税款所属时期": period,
            "入（退）库日期": date,
            "实缴（退）金额": amt
        })
    return out


def random_invoice():
    taxpayer = random_chinese_name(4) + random.choice(["科技有限公司", "信息技术公司", "电子商务有限公司", "农业科技有限公司"])
    content = {
        "编号": generate_bianhao(),
        "页码": {"currentPage": "1", "totalPage": "1"},
        "税务机关": random_chinese_name(4) + "县税务局",
        "填发日期": random_date(2023, 2025),
        "纳税人名称": taxpayer,
        "纳税人识别号": "".join(random.choices(string.digits, k=18)),
        "事项": random.choice(["增值税缴纳", "企业所得税补缴", "印花税缴纳"]),
        "填票人": random_chinese_name(2),
        "金额合计": ["伍万零捌拾贰元陆角玖分", random_money(1000, 100000)],
    }

    table = tex_like_table_rows()

    others = ["本凭证不作纳税人记账、抵扣凭证", "第1页，总共1页"]

    return {
        "title": "中华人民共和国税收完税证明",
        "content": content,
        "table": table,
        "others": others
    }


def generate_mock_data(n_files=2):
    data_list = []
    for i in range(n_files):
        filename = f"mock_{uuid.uuid4().hex[:8]}.pdf"
        n_pages = random.randint(1, 6)
        results = []
        for p in range(1, n_pages + 1):
            inv = random_invoice()
            inv["content"]["页码"]["currentPage"] = str(p)
            inv["content"]["页码"]["totalPage"] = str(n_pages)
            results.append(inv)
        data_list.append({
            "filename": filename,
            "results": results
        })

    code = random.choices([200, 500, 401, 402], weights=[90, 4, 3, 3], k=1)[0]
    message = "successful" if code == 200 else random.choice(["backend error", "stamp missing", "stamp mismatch"]) 
    return {
        "code": code,
        "message": message,
        "data": data_list
    }
import random
import string
import datetime
import uuid


def random_date(start_year=1990, end_year=2025):
    start = datetime.date(start_year, 1, 1)
    end = datetime.date(end_year, 12, 31)
    delta = end - start
    return (start + datetime.timedelta(days=random.randint(0, delta.days))).strftime("%Y-%m-%d")


def random_chinese_name(length=3):
    chars = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯昝管卢莫经房裘缪干解应宗丁宣贲邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊於惠甄魏加封任鲁顾孟"
    return "".join(random.choice(chars) for _ in range(length))


def random_money(min_value=100.0, max_value=100000.0):
    amount = round(random.uniform(min_value, max_value), 2)
    return f"¥{amount:,.2f}"


def tex_like_table_rows():
    rows = [
        ("增值税(滞纳金)", "2023-05-01 至 2023-05-31", "2024-03-05", "¥3,097.85"),
        ("增值税", "2023-05-01 至 2023-05-31", "2024-03-05", "¥23,468.54"),
        ("企业所得税", "2023-01-01 至 2023-12-31", "2024-05-09", "¥20,424.63"),
        ("企业所得税", "2024-01-01 至 2024-03-31", "2024-04-12", "¥9,208.57"),
        ("城市维护建设税", "2023-05-01 至 2023-05-31", "2024-03-05", "¥821.40"),
        ("城市维护建设税(滞纳金)", "2023-05-01 至 2023-05-31", "2024-03-05", "¥108.42"),
        ("印花税", "2023-10-01 至 2023-12-31", "2024-01-10", "¥9,071.31"),
        ("印花税", "2024-01-01 至 2024-03-31", "2024-04-10", "¥8,488.76"),
        ("印花税", "2024-04-01 至 2024-06-30", "2024-07-10", "¥9,280.19"),
        ("印花税", "2024-07-01 至 2024-09-30", "2024-10-09", "¥6,375.57"),
        ("教育费附加", "2023-05-01 至 2023-05-31", "2024-03-05", "¥352.03"),
        ("地方教育附加", "2023-05-01 至 2023-05-31", "2024-03-05", "¥234.68"),
    ]
    k = random.randint(1, len(rows))
    chosen = random.sample(rows, k)
    out = []
    for t, period, date, amt in chosen:
        out.append({
            "税种": t,
            "税款所属时期": period,
            "入（退）库日期": date,
            "实缴（退）金额": amt
        })
    return out


def generate_bianhao():
    if random.random() < 0.5:
        return 'No. ' + ''.join(random.choices(string.digits, k=18))
    else:
        prefix_options = ['25（0116）证明 ', '26（0210）证明 ', '30（1104）证明 ', '28（0302）证明 ']
        prefix = random.choice(prefix_options)
        return prefix + ''.join(random.choices(string.digits, k=9))


def random_invoice():
    taxpayer = random_chinese_name(4) + random.choice(["科技有限公司", "信息技术公司", "电子商务有限公司", "农业科技有限公司"])
    content = {
        "编号": generate_bianhao(),
        "页码": {"currentPage": "1", "totalPage": "1"},
        "税务机关": random_chinese_name(4) + "县税务局",
        "填发日期": random_date(2023, 2025),
        "纳税人名称": taxpayer,
        "纳税人识别号": "".join(random.choices(string.digits, k=18)),
        "事项": random.choice(["增值税缴纳", "企业所得税补缴", "印花税缴纳"]),
        "填票人": random_chinese_name(2),
        "金额合计": ["伍万零捌拾贰元陆角玖分", random_money(1000, 100000)],
    }
    table = tex_like_table_rows()
    others = ["本凭证不作纳税人记账、抵扣凭证", "第1页，总共1页"]
    return {
        "title": "中华人民共和国税收完税证明",
        "content": content,
        "table": table,
        "others": others
    }


def generate_mock_data(n_files=2):
    data_list = []
    try:
        for i in range(n_files):
            filename = f"mock_{uuid.uuid4().hex[:8]}.pdf"
            n_pages = random.randint(1, 6)
            results = []
            for p in range(1, n_pages + 1):
                inv = random_invoice()
                inv["content"]["页码"]["currentPage"] = str(p)
                inv["content"]["页码"]["totalPage"] = str(n_pages)
                inv["others"] = ["本凭证不作纳税人记账、抵扣凭证", f"第{p}页，总共{n_pages}页"]

                # assign per-page status code: majority 200, some 500/401/402
                page_code = random.choices([200, 500, 401, 402], weights=[60, 5, 17.5, 17.5], k=1)[0]
                # optional per-page message for non-200
                if page_code == 200:
                    page_message = "successful"
                elif page_code == 500:
                    page_message = "backend error"
                elif page_code == 401:
                    page_message = "stamp missing"
                else:
                    page_message = "stamp mismatch"

                # attach status info to each result item
                inv["status"] = {"code": page_code, "message": page_message}

                results.append(inv)
            data_list.append({
                "filename": filename,
                "results": results
            })

        # overall success
        return {
            "code": 200,
            "message": "successful",
            "data": data_list
        }
    except Exception as e:
        # in case of any unexpected error, return 500
        return {
            "code": 500,
            "message": f"backend error: {str(e)}",
            "data": []
        }
