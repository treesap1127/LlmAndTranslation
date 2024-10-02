import os
# import mysql.mysql_data as mysql_data
import llmAssist.fileReader as fileReader
# import torch

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders.unstructured import UnstructuredFileLoader



from config import conf

# CHROMA_FIEL_PATH = './chromadb/file'
CHROMA_DB_PATH = './chromadb/data'

docCount = conf['docCount']

# 파일 임베딩
async def embed_text(result_dict):
    file_name = result_dict.get('file_name')
    file_ext = result_dict.get('file_ext')
    file_data = result_dict.get('file_data')
    app_path = result_dict.get('app_path')
    user_input = result_dict.get('user_input')
    #폴더 생성
    os.makedirs(app_path+"/.cache/files", exist_ok=True)
    os.makedirs(app_path+"/.cache/embeddings", exist_ok=True)
    file_path = f"{app_path}/.cache/files/{file_name}"

    print('임베딩 시작') 

    # 버퍼의 내용 변환
    with open(file_path, "wb") as f:
        f.write(file_data)

    # 단락 -> 문장 -> 단어 순서로 재귀적으로 분할
    # text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    #     chunk_size=2000,
    #     chunk_overlap=0,
    # )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        separators=["\n\n", "\n", "(?<=\. )", " ", ""],
        length_function=len,
    )
        
    result = False

    # 파일 Document화
    if file_ext.lower() in {'pptx','txt'}:#,'pdf', 'xlsx',
        loader = UnstructuredFileLoader(file_path)

        #1.UnstructuredFileLoader로 사용 할 수 있는 확장자를 늘린다.

        # loader.load()
        docs = loader.load_and_split(text_splitter=text_splitter)
        if len(docs) < docCount:
            result = 'small'
        else : 
            result = 'True'

    # UnstructuredFileLoader로 Document화가 안되는 파일 목록
    elif file_ext.lower() in {'xlsx','xls','ppt','pdf','hwp','hwpx','doc','docx'} :
        docs = document_create(file_path, text_splitter)

        if len(docs) < docCount:
            result = 'small'
        else :
            result = 'True'
    else :
        result = 'False'
        pass

    #벡터 DB 검색기 객체 생성 
    # chroma_db = Chroma.from_documents(documents=docs, embedding=ollama_emb)

    # 벡터 DB 검색기 객체 수정 후 생성
    # ids = [str(i)+task_id for i in range(1, len(docs) + 1)]
    # try:
    #     chroma_db = Chroma(embedding_function=ollama_emb, persist_directory=CHROMA_DB_PATH , ids=ids)
    #     chroma_db.upsert(docs, embedding=ollama_emb, persist_directory=CHROMA_DB_PATH)#추가 또는 수정 <- 베타 기능
    # except Exception as e:
    #     chroma_db = Chroma.from_documents(documents=docs, embedding=ollama_emb, persist_directory=CHROMA_DB_PATH , ids=ids) 

#    ReRank # 인터넷 연결 이슈로 인해 중단
    # compressor = CohereRerank(model="rerank-multilingual-v3.0")
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, 
    #     base_retriever=retriever
    # )


    print('임베딩 및 Chroma retriever 추출 완료')
    return docs, result

# async def embed_img (file_data, file_name, file_ext, app_path, user_input):
#     # 폴더 생성
#     os.makedirs(app_path+"/.cache/files", exist_ok=True)
#     os.makedirs(app_path+"/.cache/embeddings", exist_ok=True)
#     os.makedirs(app_path+"/.cache/parquet", exist_ok=True)
#     file_path = f"{app_path}/.cache/files/{file_name}"

#     print('임베딩 시작') 

#     # 버퍼의 내용 변환
#     with open(file_path, "wb") as f:
#         f.write(file_data)

#     # 단락 -> 문장 -> 단어 순서로 재귀적으로 분할
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=500,
#         chunk_overlap=100,
#         separators=["\n\n", "\n", "(?<=\. )", " ", ""],
#         length_function=len,
#     )
    
#     # ollama 임베딩 모델 정의
#     ollama_emb = getEmbLlm()
#     result = False
#     if file_ext.lower() in {'jpg','jpeg','png','bmp','gif'} :
#         #Chroma BaseUrl로 Base64 가져옴 *encode_image(uri)add_images
#         docs= []
#         # response = requests.get('http://newgjue.mayeye.net/attach/image/eae23a0cde292eb409d118528e62d6ae/c2bce75ca13575add9491df5855789f5')
#         #TODO : test용 이미지를 링크로 적용 해두었습니다.
#         response = requests.get('https://flexible.img.hani.co.kr/flexible/normal/960/960/imgdb/resize/2019/0121/00501111_20190121.JPG')
#         image_file = BytesIO(response.content)
#         encoded_image = base64.b64encode(image_file.getvalue()).decode('utf-8')

#         metadata_data = {"source": file_path, "type": "Document"}
#         doc = Document(page_content=encoded_image, metadata=metadata_data)
#         docs.append(doc)
#         chroma_db = Chroma.from_documents(embedding=ollama_emb, documents=docs)
#         # chroma_db = Chroma.encode_image(uri=,self=chroma_db)
#         result = 'True'
#         # score = chroma_db.similarity_search_with_score(user_input)
        
#         # for doc, score_num in score:
#         #     print(f"문서에 따른 질문 점수 : {score_num}")
#         # print('문서 나뉜 수 : '+str(len(docs)))
#         # if len(docs) < docCount:
#         #     result = 'small'
#         # else :
#             # result = 'True'
#     else :
#         result = 'False'
#         pass

#     retriever = chroma_db.as_retriever()

#     return retriever, result

#DataBase 내용 임베딩
# async def embed_db (task_id):

#     # raw_text = mysql_data.annualLeave()

#     # processed_texts = []
#     # for item in raw_text:
#     #     if isinstance(item, str):
#     #         # 문자열은 그대로 추가
#     #         processed_texts.append(item)
#     #     elif isinstance(item, tuple):
#     #         # 튜플을 리스트로 변환한 후 JSON 문자열로 변환하여 추가
#     #         json_str = json.dumps(list(item), ensure_ascii=False)
#     #         processed_texts.append(json_str)
#     #     else:
#     #         raise TypeError("raw_text 리스트의 모든 요소는 문자열 또는 튜플이어야 합니다.")
#     # # 리스트의 모든 요소를 하나의 문자열로 결합
#     # raw_text = ' '.join(processed_texts)

#     # # 단락 -> 문장 -> 단어 순서로 재귀적으로 분할
#     # text_splitter = RecursiveCharacterTextSplitter(
#     #     chunk_size=500,
#     #     chunk_overlap=50,
#     #     separators=["\n\n", "\n", "(?<=\. )", " ", ""],
#     #     length_function=len,
#     # )

#     # docs = text_splitter.split_text(raw_text)

#     # docs_json = json.dumps(docs, ensure_ascii=False, indent=4)

#     ollama_emb = getEmbLlm()

#     # 벡터 DB 저장 시 아래 코드 주석 해제
#     # ids = [str(i)+task_id for i in range(1, len(docs) + 1)]
#         # CHROMA 저장
#     # try:
#     chroma_db = Chroma(embedding_function=ollama_emb)
#     # chroma_db = Chroma(embedding_function=ollama_emb, persist_directory=CHROMA_DB_PATH , ids=ids)
#         # chroma_db.upsert(docs, embedding=ollama_emb, persist_directory=CHROMA_DB_PATH)#추가 또는 수정 <- 베타 기능
#     # except Exception as e:
#         # chroma_db = Chroma.from_texts(texts=docs, embedding=ollama_emb, persist_directory=CHROMA_DB_PATH , ids=ids)        
#     # chroma_db = Chroma.from_texts(texts=docs, embedding=ollama_emb, ids=ids) #벡터 DB 객체 get

#     retriever = dbChanger(chroma_db,1,10,1.0)
#     print('임베딩 및 Chroma retriever 추출 완료')

#     return retriever, chroma_db

# fileData -> langChain.Document 변경
def document_create(file_path, text_splitter):
    file_extension = file_path.split('.')[-1].lower()

    if file_extension == 'xlsx':
        text_list, metadata_list = fileReader.read_xlsx(file_path)
    elif file_extension == 'xls':
        text_list, metadata_list = fileReader.read_xls(file_path)        
    elif file_extension == 'pdf':
        text_list, metadata_list = fileReader.read_pdf(file_path)
    elif file_extension == 'ppt':
        text_list, metadata_list = fileReader.read_ppt(file_path)
    elif file_extension == 'hwp':
        text_list, metadata_list = fileReader.read_hwp(file_path)
    elif file_extension == 'hwpx':
        text_list, metadata_list = fileReader.read_hwpx(file_path)
    elif file_extension == 'doc':
        text_list, metadata_list = fileReader.read_doc(file_path)
    elif file_extension == 'docx':
        text_list, metadata_list = fileReader.read_docx(file_path)

    assert len(text_list) == len(metadata_list), "텍스트와 메타데이터 리스트의 길이가 같아야 합니다."

    # docs = []
    split_documents = []

    #Document화 
    for content, metadata in zip(text_list, metadata_list):
        splits = text_splitter.split_text(content)
        for split in splits:
            split_doc = Document(page_content=split, metadata=metadata)
            split_documents.append(split_doc)

    # 분할된 문서를 사용할 경우 split_documents 사용
    return split_documents


