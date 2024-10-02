import os
import uvicorn
from fastapi import FastAPI, APIRouter, WebSocket, WebSocketDisconnect

from streaming import StreamFileResponse, StreamResponse
from uuid import uuid4
import logging.config
from config import conf

#환경변수 설정
os.environ['KMP_DUPLICATE_LIB_OK']='True'#← OMP오류... 대처 방안 
os.environ["OPENAI_API_KEY"] = 'ollama' #llm load
os.environ['LANGCHAIN_TRACING_V2']='true' # langChain 중간 결과값 가져오기
os.environ['LANGCHAIN_API_KEY']=''#LANGSMITH API 키
os.environ["LANGCHAIN_ENDPOINT"] = 'https://api.smith.langchain.com'#LANGSMITH 주소
os.environ['LANGCHAIN_PROJECT']='agentTest' #LANGSMITH 프로젝트 정보
os.environ['AGENTOPS_API_KEY']='' # crowai AGENTOPS 미 연동 시 오류 발생

app = FastAPI()
router = APIRouter()

# 개발 서버 실행 코드 [ /www/Python/venv/bin/gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 192.168.10.41:8000 --log-config /www/Python/venv/logs/uvicorn_log.ini ]
# 로컬 서버 실행 코드 [uvicorn main:app --host 127.0.0.1 --port 8000 --workers 2] or main.py->F5

import uvicorn
if __name__ == "__main__":
    if conf['os'] == 'linux': 
        uvicorn.run(app, host="192.168.10.41", port=8000)
        logging.config.fileConfig('/www/Python/venv/logs/uvicorn_log.ini')
if conf['os'] == 'window': 
    app_path = 'C:/Python'
elif conf['os'] == 'linux': 
    app_path = '/www/Python'

#폴더 생성
os.makedirs(app_path+"/chromadb/data", exist_ok=True)
os.makedirs(app_path+"/chromadb/file", exist_ok=True)

EXCEPTION_EXT = {'xls','xlsx','txt','hwp','hwpx','doc','docx','pdf','ppt','pptx'}
IMG_EXT = {'jpg','jpeg','png','bmp','gif'}

# 임베딩 폴더 생성
# print("현재 작업 디렉토리:", os.getcwd())

@router.websocket('/ws')
async def chat(websocket: WebSocket):
    task_id = str(uuid4()) # 콜백 요청에 대한 구분값
    os.makedirs(app_path+"/chromadb/data/"+task_id, exist_ok=True)
    file_name = ''
    await websocket.accept()
    try:
        data = await websocket.receive_bytes()
        result_dict = {}

        # 웹소켓 데이터 분류 대략 20 거래일
        if b'\0' in data:# 구분 값이 있을 경우
            user_input_sep = data.find(b'\0',0)
            user_input = data[:user_input_sep].decode('utf-8')
            file_name_index = data.find(b'\0',user_input_sep+1)
            file_name = data[user_input_sep+1:file_name_index].decode('utf-8')
            file_ext = file_name.split('.')[-1].lower().replace('.','')
            file_data = data[file_name_index+1:]
            result_dict = {
                'user_input': user_input,
                'file_name': file_name,
                'file_ext': file_ext,
                'file_data': file_data,
                'app_path' : app_path,
                'task_id' : task_id
            }

        print('데이터 추출 완료')

        # 문서 임베딩 및 답변
        if file_name and file_ext and file_ext.lower() in EXCEPTION_EXT:
            await StreamFileResponse(websocket, result_dict)
            
        
        # 이미지 임베딩 및 답변 <- 많은 수정으로 인해 임시 보류
        # elif file_name and file_ext and file_ext.lower() in IMG_EXT:
        #     # retriever = await embedding.embed_img(file_data, file_name, file_ext, app_path)
        #     #await StreamImgResponse(websocket, user_input, retriever, task_id)
        #     retriever, result = await embedding.embed_img(file_data, file_name, file_ext, app_path, user_input)
        #     await StreamFileResponse(websocket, user_input, retriever, task_id)
        #     if result == 'False':
        #         await websocket.send_text('파일 읽기에 실패 했습니다.')
        #         await websocket.close()
        #     elif result == 'small':
        #         await websocket.send_text('문서 및 prompt 내용이 너무 적어 요약 할 수 없습니다.')
        #         await websocket.close()
        #     else :
        #         await StreamFileResponse(websocket, user_input, retriever, task_id)
        #     pass 
        # 텍스트 답변
        else :
            user_input = data.decode('utf-8')
            await StreamResponse(websocket, user_input, task_id)

    except WebSocketDisconnect:
        print("연결 해제")

app.include_router(router)

# 로컬 Test용  
if  __name__ == "__main__":
   if conf['os'] == 'window' :
        uvicorn.run(app, host="127.0.0.1", port=8001)


