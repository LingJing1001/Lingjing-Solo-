import pypdf
reader = pypdf.PdfReader(r'F:\weixin\xwechat_files\eric05001_62a4\msg\file\2026-10\搜索算法和胜利检测(1).pdf')
for i, page in enumerate(reader.pages):
    print(f'=== Page {i+1} ===')
    print(page.extract_text() or '')
