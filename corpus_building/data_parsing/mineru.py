import os, uuid, argparse, json
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
from datetime import datetime
from .utils import download_json, download_and_modify_json



def process_single_pdf(pdf_file_name, output_dir, image_subdir, simple_output):
    """
    处理PDF文件，将其转换为Markdown格式并保存相关资源
    :param pdf_file_name: PDF文件绝对路径
    :param output_dir: 输出根目录，接收用户输入
    :param image_subdir: 图片子目录名，默认为'images'
    :param simple_output: 是否使用简单输出模式
    """
    # 从PDF绝对路径获取不带后缀的文件名
    name_without_suff = os.path.splitext(os.path.basename(pdf_file_name))[0]
    # 生成唯一标识符
    # unique_id = uuid.uuid4()
    # 创建输出的子目录名
    # output_subdir = f"{name_without_suff}-{unique_id}"
    output_subdir = f"{name_without_suff}"
    # 构建图片存储目录和md文件的路径名
    local_image_dir = os.path.join(output_dir, output_subdir, image_subdir)
    local_md_dir = os.path.join(output_dir, output_subdir)
    # 创建目录
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)

    # 创建文件写入器
    image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
    # 创建文件读取器并读取PDF文件
    reader1 = FileBasedDataReader("")
    pdf_bytes = reader1.read(pdf_file_name)

    # 创建数据集对象
    ds = PymuDocDataset(pdf_bytes)
    # 根据PDF类型选择处理方式
    if ds.classify() == SupportedPdfParseMethod.OCR:
        # 使用OCR模式处理
        infer_result = ds.apply(doc_analyze, ocr=True)
        pipe_result = infer_result.pipe_ocr_mode(image_writer)
    else:
        # 使用文本模式处理
        infer_result = ds.apply(doc_analyze, ocr=False)
        pipe_result = infer_result.pipe_txt_mode(image_writer)

    # 构建md文件的完整路径
    md_file_path = os.path.join(os.getcwd(), local_md_dir, f"{name_without_suff}.md")
    abs_md_file_path = os.path.abspath(md_file_path)

    if simple_output:
        # 简单输出模式，只输出md和内容列表
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", os.path.basename(local_image_dir))
        pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json",
                                      os.path.basename(local_image_dir))
        return local_md_dir
    else:
        # 完整输出模式，输出所有处理结果
        pipe_result.dump_md(md_writer, f"{name_without_suff}.md", os.path.basename(local_image_dir))
        pipe_result.dump_content_list(md_writer, f"{name_without_suff}_content_list.json",
                                      os.path.basename(local_image_dir))

    # 生成可视化文件
    infer_result.draw_model(os.path.join(local_md_dir, f"{name_without_suff}_model.pdf"))
    pipe_result.draw_layout(os.path.join(local_md_dir, f"{name_without_suff}_layout.pdf"))
    pipe_result.draw_span(os.path.join(local_md_dir, f"{name_without_suff}_spans.pdf"))

    # return local_md_dir



def merge_corpus(CORPUS_PATH, OUTPUT_FILE):
    all_instances = []

    json_files = sorted(CORPUS_PATH.rglob("*.json"))  
    for book_idx, json_file in enumerate(json_files):
        with open(json_file, "r", encoding="utf-8") as f:
            try:
                items = json.load(f)          
                if isinstance(items, list):
                    for inst in items:
                        inst["book_idx"] = book_idx
                    all_instances.extend(items)
                else:
                    print(f"[Warn] {json_file} 不是列表，已跳过")
            except json.JSONDecodeError as e:
                print(f"[Error] {json_file} 解析失败: {e}")

    print(f"Total merged instances: {len(all_instances)}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_instances, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved to {OUTPUT_FILE}")



def parse_pdfs(file_root: str,  # 知识库文件根目录, 解析该路径下的所有文件
               # parse_output_dir: str, # 解析结果保存根目录, 即merge_corpus的第一个输入变量, 直接默认file_root下面创建一个文件夹就行了
               output_dir: str, # 合并后的最终解析结果保存目录, 即merge_corpus的第二个输入变量,
               image_dir: str = "/data/huali_mm/images", # 媒体文件保存根目录
               config_dir: str = "magic-pdf.json", # 配置文件路径
               simple_output: bool = True
              )

    # 下载并修改配置json文件
    # json_url = 'https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/magic-pdf.template.json'
    # config_file_name = 'magic-pdf.json'
    # config_file = os.path.join(os.path.expanduser('~'), config_file_name)
    # json_mods = {
    #     'models-dir': "cuda",
    # }
    # download_and_modify_json(json_url, config_file, json_mods)

    file_list = [f for f in os.listdir(file_root) if f.lower().endswith('.pdf')]
    # parse_file_list = []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    parse_output_dir = os.path.join(file_root, f'parsed_{timestamp}')
    os.makedirs(parse_output_dir, exist_ok=True)
    for file in file_list:
        file_dir = process_single_pdf(pdf_file_name = os.path.join(file_root, file),
                                      output_dir    = parse_output_dir,
                                      image_subdir  = image_dir,
                                      simple_output = simple_output)
        # parse_file_list.append(file_dir)
        
    merge_corpus(parse_output_dir, output_dir)
    
