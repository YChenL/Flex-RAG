import os
import uuid
from magic_pdf.data.data_reader_writer import FileBasedDataWriter, FileBasedDataReader
from magic_pdf.data.dataset import PymuDocDataset
from magic_pdf.model.doc_analyze_by_custom_model import doc_analyze
from magic_pdf.config.enums import SupportedPdfParseMethod
import argparse
import json
import requests

def download_json(url):
    # 下载JSON文件
    response = requests.get(url)
    response.raise_for_status()  # 检查请求是否成功
    return response.json()


def download_and_modify_json(url, local_filename, modifications):
    if os.path.exists(local_filename):
        data = json.load(open(local_filename))
        config_version = data.get('config_version', '0.0.0')
        if config_version < '1.2.0':
            data = download_json(url)
    else:
        data = download_json(url)

    # 修改内容
    for key, value in modifications.items():
        data[key] = value

    # 保存修改后的内容
    with open(local_filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)




def process_pdf(pdf_file_name, output_dir, image_subdir="images", simple_output=False):
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

    return local_md_dir


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(description='PDF文件解析工具：对PDF进行解析')
    
    # 路径参数
    parser.add_argument('pdf_file', type=str, help='要处理的PDF文件路径（绝对路径）')
    parser.add_argument('output_dir', type=str, help='输出结果的根目录路径')
    parser.add_argument('--config_dir', type=str, default="magic-pdf.json",
                       help='配置文件路径（默认: magic-pdf.json）')
    
    # PDF解析参数
    parser.add_argument('--image_dir', type=str, default="images",
                       help='图片存储的子目录名（默认: /images）')
    parser.add_argument('--simple_output', type=bool, default=False,
                       help='是否启用简单输出模式（仅输出文本内容和解析出的图片）')
    
    args = parser.parse_args()


    # 下载并修改配置json文件
    json_url = 'https://gcore.jsdelivr.net/gh/opendatalab/MinerU@master/magic-pdf.template.json'
    config_file_name = 'magic-pdf.json'
    config_file = os.path.join(os.path.expanduser('~'), config_file_name)
    json_mods = {
        'models-dir': "cuda",
    }
    download_and_modify_json(json_url, config_file, json_mods)


    # 调用处理函数
    result_path = process_pdf(
        pdf_file_name=args.pdf_file,
        output_dir=args.output_dir,
        image_subdir=args.image_dir,
        simple_output=args.simple_output
    )
    
    print(f"PDF文件解析结果已存储在: {result_path}")


if __name__ == "__main__":
    main()



