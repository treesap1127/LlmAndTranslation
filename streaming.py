import asyncio
import torch
import llmAssist.promptInit as promptInit
from llmAssist.streaming_callback import StreamingCallback
from llmAssist.text_splitter import embed_text
from config import conf
from fastapi import WebSocketDisconnect
from langchain_core.output_parsers.json import SimpleJsonOutputParser

from model.model_setup import getLlm, getEmbLlm, getOriLlm, getAssiLlm
# from model.model_setup import getAgentLlm, getAgentAnswerLlm

#from langchain.chains import RetrievalQA, StuffDocumentsChain, LLMSummarizationCheckerChain
from langchain.chains.retrieval_qa.base import RetrievalQA
# from langchain.chains.combine_documents.stuff import StuffDocumentsChain
#from langchain.chains.llm_summarization_checker.base import LLMSummarizationCheckerChain
from langchain.chains.summarize import load_summarize_chain

from langchain_chroma import Chroma
# from langchain_community.vectorstores import Chroma,FAISS

from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.memory import ConversationBufferMemory

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, LLMChain

from raptor.clusterEmbedding import recursive_embed_cluster_summarize

#from llama_index.core.postprocessor import SentenceTransformerRerank

# agent Module
# from crewai import Agent, Task, Crew 
# import agentops
# from crewai_tools import PDFSearchTool, WebsiteSearchTool, SerperDevTool, RagTool
# from langchain.tools import tool
# from langchain.agents import create_openai_tools_agent, create_tool_calling_agent
# from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
# from langchain.agents.format_scratchpad.openai_tools import (format_to_openai_tool_messages)
# from langchain.agents import create_openai_functions_agent
# from langchain.agents import AgentExecutor
# from langchain.tools.retriever import create_retriever_tool
# from crewai.tools.agent_tools import AgentTools
# from llmAssist.retrieverTool import GetRetriever
# from langchain.callbacks.manager import CallbackManager #콜백
# search 
# from langchain_community.tools import DuckDuckGoSearchRun

EMB_MODEL_NM = conf['embModelNm']

EMB_LANGSERVE_ADD = conf['embLangserveAdd']

async def StreamFileResponse(websocket, result_dict):
    task_id = result_dict.get('task_id')
    app_path = result_dict.get('app_path')
    streaming_callback = StreamingCallback(task_id,websocket)        
    # llm = getLlm()
    llm = getLlm(streaming_callback)
    # OriLlm = getOriLlm()
    # AssiLlm = getAssiLlm()
    # AgentLlm = getAgentLlm()

    user_input = result_dict.get('user_input')
    docs, result= await embed_text(result_dict)
    # 반복 검색 시 사용
    retriever = docsChanger(docs, 15, 30, 0.1, user_input, app_path+'/chromadb/data',task_id) #db(vector DB), k(적용 문서 수), fetch_k(총 문서 수), labda_mult(다양성)
 
    FEW_SHOT_CHAIN_PROMPT = promptInit.FewShotPrompt()    
    # ASSI_PROMPT = promptInit.ASSI_PROMPT_TEMPLATE

    #crewAi Prompt
    # ASSI_DESCRIPTION = promptInit.ASSI_DESCRIPTION
    # AGENT_GOAL = promptInit.AGENT_GOAL
    # ASSI_GOAL = promptInit.ASSI_GOAL
    # AGENT_BACKSTORY = promptInit.AGENT_BACKSTORY
    # ASSI_BACKSTORY = promptInit.ASSI_BACKSTORY
    # AGENT_EXPECTED_OUTPUT = promptInit.AGENT_EXPECTED_OUTPUT
    # ASSI_TASK_EXPECTED_OUTPUT = promptInit.ASSI_TASK_EXPECTED_OUTPUT
    


    if result == 'False':
        await websocket.send_text('파일 읽기에 실패 했습니다.')
        await websocket.close()
    elif result == 'small':
        await websocket.send_text('문서 및 prompt 내용이 너무 적어 요약 할 수 없습니다.')
        await websocket.close()
    else :
        ############################################################################
                        #검색기 요약 체인 (주제별 내용 요청)
        

        # docs=retriever.invoke(user_input)

        # chain = load_summarize_chain(OriLlm, chain_type="stuff") # 문서 내용 요약 체인

        # print('assi send 중...')

        # data = chain.invoke({"input_documents": docs})
        # data_text = data.get('output_text')

        # print(data_text)
        print('###################문서 데이터 답변 완료####################################')
        #############################################################################
        #                 # 검색 결과 메모리 저장
        # memory = ConversationBufferMemory(memory_key=task_id)

        # memory.chat_memory.add_user_message('문서 내용')
        # memory.chat_memory.add_ai_message(data_text)
        try :
        ################################################################################
                            #메모리 반복 체인
            # itreatorchain =  retriever | FEW_SHOT_CHAIN_PROMPT | AssiLlm  #memory 추가 불가

            # # for i in range(3) :
            # user_message = ''
            # # if i==0 :
            #     # user_message = ITREATOR_PROMPT_1
            # itreator_result = itreatorchain.invoke(user_input) 
            # # if i==1 :
            # #     user_message = ITREATOR_PROMPT_2
            # #     itreator_result = itreatorchain.invoke(user_message)
            # # if i==2 :
            # #     user_message = ITREATOR_PROMPT_3
            # #     itreator_result = itreatorchain.invoke(user_message)
            # # trans = Translate(to_lang='korean')
            # print(itreator_result)
            # print('###################0차 답변 완료####################################')
            # # print('###################'+str(i+1)+'차 답변 완료####################################')
            # # from_chain_type의 메모리는 자동 저장 
            # memory.chat_memory.add_user_message(user_message)        
            # memory.chat_memory.add_ai_message(itreator_result.content)

        ################################################################################ 
                        #메모리를 추가를 포함한 모든 내용 체인으로 답변
                
            # # prompt 생성
            # LAST_CHAIN_PROMPT = promptInit.MemoryPrompt() 

            # #답변 체인 생성
            # lastchain = RetrievalQA.from_chain_type(
            #             memory=memory,
            #             retriever=retriever,
            #             # chain_type_kwargs={"prompt": FEW_SHOT_CHAIN_PROMPT},
            #             chain_type_kwargs={"prompt": LAST_CHAIN_PROMPT},
            #             llm=llm,
            #         )
            # combine_docs_chain = create_stuff_documents_chain( # memory 추가 불가
            #     llm, FEW_SHOT_CHAIN_PROMPT
            # )
            # lastchain = create_retrieval_chain(retriever, combine_docs_chain )
            lastchain = retriever | FEW_SHOT_CHAIN_PROMPT | llm   #memory 추가 불가 ( retriever,llm 순서 중요 )
            # print(lastchain.invoke(user_input))#

            # ai from_chain_type 실행
            async def stream_process_start():
                async for _ in lastchain.astream(user_input): # RetrievalQA.from_chain_type
                # async for _ in lastchain.astream({"input":user_input}): # create_retrieval_chain
                            # FewShotPrompt와 체인에서 요구하는 필드가 다름 (불가능)
                    pass
            # 저장 된 ai 답변 출력 task 실행
            async def stream_response(task):
                try:
                    async for token in streaming_callback.token_generator():
                        await websocket.send_text(token)
                except WebSocketDisconnect:
                    #답변 정지 시 task 정지
                    task.cancel()
            task = asyncio.create_task(stream_process_start())
            await stream_response(task)

            # await stream_response()
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            pass

    # ############################################################################


#     ##########################################
    #     #agent 구성 (Crew Ai 사용)
    #     #최신 데이터 서칭 결과 포함 관련 함수
    #     def ddgsearch(question: str) -> str:
    #         # Function logic here
    #         return DuckDuckGoSearchRun().run(question)

    #     searchTool = WebsiteSearchTool(website='https://www.mayeye.net/kor/solution')
    #     test=ddgsearch('https://www.mayeye.net/kor/solution')
    #     pdfTool = PDFSearchTool(
    #     pdf = 'D:\업로드 테스트 파일\확장자\메이아이_사용설명서.pdf'
    #     ,
    #     config = dict(
    #     #     llm=dict(
    #     #         provider="ollama", # or google, openai, anthropic, llama2, ...
    #     #         # api_key="ollama",
    #     #         # api_base = 'http://192.168.10.41:11434/v1' , 
    #     #         config=dict(#openapi 어떤 config 매개변수 전송 해야하는지 모르겠네
    #     #             base_url=EMB_LANGSERVE_ADD,
    #     #             model="bnksys/yanolja-eeve-korean-instruct-10.8b",
    #     #         ),
    #     #     ),
    #         embedder=dict(
    #             provider="ollama", # or openai, ollama, ...
    #             config=dict(
    #                 model=EMB_MODEL_NM,
    #                 base_url = EMB_LANGSERVE_ADD,
    #             ),
    #         ),
    #     )
    # )
    #     tools= []
    #     tools.append(WebsiteSearchTool('https://www.mayeye.net/kor/solution'))
    #     # retriever_tool = GetRetriever(retriever = retriever, query = user_input)
    #     retriever_tool = create_retriever_tool(
    #         retriever,
    #         name="data_search",
    #         description="요약해야하는 정보를 PDF 문서에서 검색합니다. '문서' 와 관련된 질문은 이 도구를 사용해야 합니다!",
    #     )
    #     tools.append(retriever_tool)

    #     # print(tool)    
    #     researcher = Agent(
    #         role='Researcher',
    #         goal=AGENT_GOAL, 
    #         backstory=AGENT_BACKSTORY,
    #         verbose=True,           # 로깅
    #         allow_delegation=True, # 다른 Agent에게 작업 위임 여부
    #         tools=tools,
    #         max_iter = 10,
    #         llm=getAgentLlm()           # local model
    #     )

    #     task1 = Task(
    #         agent=researcher,
    #         description=user_input,
    #         expected_output=AGENT_EXPECTED_OUTPUT,
    #     )
    #     # agent 구성 (조수 역)
    #     writer = Agent(
    #         role='assistant',#에이전트의 기능을 정의합니다. 에이전트가 가장 적합한 작업 유형을 결정합니다.
    #         goal=ASSI_GOAL,
    #         backstory=ASSI_BACKSTORY,
    #         verbose=True,            # 로깅
    #         allow_delegation=True,   # 다른 Agent에게 작업 위임 여부
    #         max_iter = 10,
    #         tools=tools, 
    #         llm=getAgentAnswerLlm()  # Using the local model
    #     )

    #     task2 = Task(
    #         agent=writer,
    #         description=ASSI_DESCRIPTION,
    #         expected_output=ASSI_TASK_EXPECTED_OUTPUT,
    #     )

    #     crew = Crew(
    #         agents=[researcher, writer],
    #         tasks=[task1, task2],
    #         verbose=2, # 로그 레벨
    #         step_callback = '',
    #         task_callback = '',
    #         tools=tools,
    #         # retriever=retriever,
    #         # memory=True, #OutOfMemory Casing존재로 일단 주석
    #         # embedder={
    #         #     "provider": "ollama",
    #         #     "config":{
    #         #             'model': EMB_MODEL_NM
    #         #             , 'base_url': EMB_LANGSERVE_ADD
    #         #         }
    #         # },
    #         # task_callback=streaming_callback
    #         # manager_callbacks=streaming_callback,
    #     )
    #     # result1 = crew.kickoff()
    #     # print(result1)
    #     # result = crew.kickoff_async(inputs={"ages": [25, 30, 35, 40, 45]})
    #     # print(result)    
    #     agentops.init()
    #     result1 = crew.kickoff()
    #     result1 =result1.raw
    #     print(result1)
    #     await websocket.send_text(result1)
    #     # agent_response = crew.execute(task)
    #     # print (agent_response)
    #     # print(result)
############################################################
    # # 응답 및 stream task 실행
    # async def stream_process_start():
    #     async for token in await crew.kickoff_for_each():
    #         await websocket.send_text(token)

    # async def stream_response():
    #     task = asyncio.create_task(stream_process_start())
    #     try:
    #         await task 
    #     except WebSocketDisconnect:
    #         task.cancel()

    # await stream_response()

    # async def stream_process_start():
    #     async for _ in await crew.kickoff_async():
    #         pass

    # async def stream_response(task):
    #             try:
    #                 async for token in streaming_callback.token_generator():
    #                     await websocket.send_text(token)
    #             except WebSocketDisconnect:
    #                 #답변 정지 시 task 정지
    #                 task.cancel()
    # task = asyncio.create_task(stream_process_start())

    # await stream_response(task)

    ###########################################


async def StreamResponse(websocket, user_input, task_id):
    streaming_callback = StreamingCallback(task_id,websocket)        
    QA_CHAIN_PROMPT = promptInit.ChatPrompt()
    
    llm = getLlm(streaming_callback)
    #ollama 모델 chain 생성
    qa_chain = QA_CHAIN_PROMPT | llm
    print('send 중...')

    #응답 및 stream task 실행
    async def stream_process_start():
        async for token in qa_chain.astream(user_input):
            await websocket.send_text(token.content)
    #task로 model 출력 조정
    async def stream_response():
        task = asyncio.create_task(stream_process_start())
        try:
            await task 
        except WebSocketDisconnect:
            task.cancel()

    await stream_response()
    
# docs -> retriever 변경
def docsChanger(docs,k,fetch_k,lambda_mult,user_input,data_path, task_id):
    ollama_emb = getEmbLlm()
    chroma_db = Chroma.from_documents(documents=docs, embedding=ollama_emb, persist_directory=data_path+'/'+task_id, ids=None)

    #######################VectorStoreRetriever###################
    base_retriever = chroma_db.as_retriever(
    # # 검색 유형을 "유사도 점수 임계값"으로 설정합니다.
    # search_type="similarity_score_threshold",
    # # 검색 인자로 점수 임계값을 0.5로 지정합니다.
    # search_kwargs={"score_threshold": 0.75},
        search_type='mmr',
        search_kwargs={'k': k
                    , 'fetch_k': fetch_k
                #    , 'lambda_mult': lambda_mult
                    },
    )# 캐시 임베딩 -> 어짜피 ollama 는 안됨 

    # data = base_retriever.invoke(user_input)
    # print (data)

    return base_retriever 

