from langchain_community.llms import Ollama 
from langchain_openai import ChatOpenAI
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
import openai
from langchain.callbacks.manager import CallbackManager #콜백

from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings

from config import conf

MODEL_NM = conf['modelNm']
ASSI_MODEL_NM = conf['assiModelNm']

LANGSERVE_ADD = conf['langserveAdd']

EMB_MODEL_NM = conf['embModelNm']

# AGENT_MODEL_NM = conf['agentModelNm']

OPEN_AI_LANGSERVE_ADD = conf['openAiLangserveAdd']
EMB_LANGSERVE_ADD = conf['embLangserveAdd']
 
def getLlm(streaming_callback):
    return ChatOpenAI(
                openai_api_key="ollama"
                , openai_api_base =OPEN_AI_LANGSERVE_ADD
                , callbacks=CallbackManager([streaming_callback])
                , model_name=MODEL_NM
                , max_tokens= 1024 #defalut 64 답변 토큰 최대 수 -1= 무한생성, -2 = 컨텐스트내에서 최대
                , streaming = True
                , frequency_penalty= 1.3# defalut 1.1 이전 토큰 반복에 대한 처벌 높을수록 같은 토큰X 1.3미만 요망
                , temperature= 0 # 단어 출력 확률을 집중하는지,  기본값 0.8 낮을수록 신뢰 mayby max 100
                , top_p= 0.1#defalut 0.9 값이 높을수록 텍스트 다양 낮을수록 정확도 향상 (이게 너무 높아서 다른 주제도 끌고 온것 + 너무 낮을 경우 답변에 중요 단어가 빠짐)
                # ex) 낮을경우 메이아이 팁 -> 메신저 알림에 대해 그냥 사용하라고만 나옴 적정 0.3
                # 근데 일부 어느정도 조절을 해야지 비슷한 단어도 잘 찾을텐데 특히 temperature
    )
def getOriLlm():
    return ChatOpenAI(
                openai_api_key="ollama"
                , openai_api_base =OPEN_AI_LANGSERVE_ADD
                # , callbacks=CallbackManager([streaming_callback])
                , model_name=MODEL_NM
                , max_tokens= 1024 #defalut 64 답변 토큰 최대 수 -1= 무한생성, -2 = 컨텐스트내에서 최대
                , streaming = True
                , frequency_penalty= 1.3# defalut 1.1 이전 토큰 반복에 대한 처벌 높을수록 같은 토큰X 1.3미만 요망
                , temperature= 0 # 단어 출력 확률을 집중하는지,  기본값 0.8 낮을수록 신뢰 mayby max 100
                , top_p= 0.1#defalut 0.9 값이 높을수록 텍스트 다양 낮을수록 정확도 향상 (이게 너무 높아서 다른 주제도 끌고 온것 + 너무 낮을 경우 답변에 중요 단어가 빠짐)
                # ex) 낮을경우 메이아이 팁 -> 메신저 알림에 대해 그냥 사용하라고만 나옴 적정 0.3
                # 근데 일부 어느정도 조절을 해야지 비슷한 단어도 잘 찾을텐데 특히 temperature
    )
def getAssiLlm():
    return ChatOpenAI(
                openai_api_key="ollama"
                , openai_api_base =OPEN_AI_LANGSERVE_ADD
                # , callbacks=CallbackManager([streaming_callback])
                , model_name=ASSI_MODEL_NM
                , max_tokens= 1024 #defalut 64 답변 토큰 최대 수 -1= 무한생성, -2 = 컨텐스트내에서 최대
                , streaming = True
                , frequency_penalty= 1.3# defalut 1.1 이전 토큰 반복에 대한 처벌 높을수록 같은 토큰X 1.3미만 요망
                , temperature= 0 # 단어 출력 확률을 집중하는지,  기본값 0.8 낮을수록 신뢰 mayby max 100
                , top_p= 0.1#defalut 0.9 값이 높을수록 텍스트 다양 낮을수록 정확도 향상 (이게 너무 높아서 다른 주제도 끌고 온것 + 너무 낮을 경우 답변에 중요 단어가 빠짐)
                # ex) 낮을경우 메이아이 팁 -> 메신저 알림에 대해 그냥 사용하라고만 나옴 적정 0.3
                # 근데 일부 어느정도 조절을 해야지 비슷한 단어도 잘 찾을텐데 특히 temperature
    )
# def getEmbLlm():
#     return HuggingFaceEndpointEmbeddings(
#         model='jhgan/ko-sroberta-multitask',
#         task="feature-extraction",
#         huggingfacehub_api_token='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',
#         # huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
#     )
def getEmbLlm():
    # from langchain_huggingface.embeddings import HuggingFaceEmbeddings

    # # model_name = "upskyy/bge-m3-korean"
    # model_name = "Suchae/bge-m3-Korean-Judgment-Transducer-Verifier-v1.1"
    # model_kwargs = {"device": "cuda"}
    # encode_kwargs = {"normalize_embeddings": True}
    # return HuggingFaceEmbeddings(
    #     model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    # )
    # from langchain_upstage import UpstageEmbeddings

    # 문장 전용 임베딩 모델
    # return UpstageEmbeddings(model="solar-embedding-1-large-passage", api_key="up_9pQPHYoF7gjdLFSovG6TTp9TQYw9Q")
    return OllamaEmbeddings(
        model=EMB_MODEL_NM
        , base_url=EMB_LANGSERVE_ADD
        #, num_ctx= 2048#defalut 2048 컨텍스트 창 설정 (질문 토큰)
        #, num_predict= 512 #defalut 64 답변 토큰 최대 수 -1= 무한생성, -2 = 컨텐스트내에서 최대

        #  , mirostat= 2 #defalut 0, 0 = 비활성화, 1 = Mirostat, 2 = Mirostat 2.0
        #  , mirostat_eta= 0.5 # defalut 0.1 #응답 속도에 따른 정확도 설정 낮을 수록 천천히 대답하고 정확도 상승
        #  , mirostat_tau= 3.0 #defalut 5.0 값이 낮을수록 텍스트가 더 집중적이고 일관성 증진 낮을 수록 다양성
        # mirostat 같은 경우 채팅의 일관성 등이 필요 할 때 사용

        #, repeat_last_n= 64#defalut 64 반복방지 n개의 토큰을 기억하고 반복 금지  128 이하
        , repeat_penalty= 1.3# defalut 1.1 이전 토큰 반복에 대한 처벌 높을수록 같은 토큰X 1.3미만 요망
        , temperature= 0 # 단어 출력 확률을 집중하는지,  기본값 0.8 낮을수록 신뢰 mayby max 100
        #, tfs_z= 2.0 #defalut 1.0 tail free Sampling <-(비활성화) 출력 가능성이 낮은 토큰 제외
        , top_k= 40#defalut 40 높을 수록 창의적 낮을수록 정확
        # ex) 낮을경우 질문에 대한 대답만 하며 높을 경우 부연 설명 추가 적정-> 40
        , top_p= 0.1#defalut 0.9 top-k와 작동 값이 높을수록 텍스트 다양 낮을수록 정확도 향상 (이게 너무 높아서 다른 주제도 끌고 온것 + 너무 낮을 경우 답변에 중요 단어가 빠짐)
        # ex) 낮을경우 메이아이 팁 -> 메신저 알림에 대해 그냥 사용하라고만 나옴 적정 0.3
        # 근데 일부 어느정도 조절을 해야지 비슷한 단어도 잘 찾을텐데 특히 temperature
    )

# def getAgentLlm():
#     return ChatOpenAI(
#                 openai_api_base =OPEN_AI_LANGSERVE_ADD
#                 , model_name=AGENT_MODEL_NM
#                 , max_tokens= 512 #defalut 64 답변 토큰 최대 수 -1= 무한생성, -2 = 컨텐스트내에서 최대
#                 , frequency_penalty= 1.3# defalut 1.1 이전 토큰 반복에 대한 처벌 높을수록 같은 토큰X 1.3미만 요망
#                 , temperature= 0.0 # 단어 출력 확률을 집중하는지,  기본값 0.8 낮을수록 신뢰 mayby max 100
#                 , top_p= 0.1#defalut 0.9 값이 높을수록 텍스트 다양 낮을수록 정확도 향상 (이게 너무 높아서 다른 주제도 끌고 온것 + 너무 낮을 경우 답변에 중요 단어가 빠짐)
#                 # ex) 낮을경우 메이아이 팁 -> 메신저 알림에 대해 그냥 사용하라고만 나옴 적정 0.3
#                 # 근데 일부 어느정도 조절을 해야지 비슷한 단어도 잘 찾을텐데 특히 temperature
#     )
# def getAgentAnswerLlm():
#     return ChatOpenAI(
#         openai_api_base =OPEN_AI_LANGSERVE_ADD
#         , model_name=MODEL_NM
#         , max_tokens= 512 #defalut 64 답변 토큰 최대 수 -1= 무한생성, -2 = 컨텐스트내에서 최대
#         , frequency_penalty= 1.3# defalut 1.1 이전 토큰 반복에 대한 처벌 높을수록 같은 토큰X 1.3미만 요망
#         , temperature= 0.0 # 단어 출력 확률을 집중하는지,  기본값 0.8 낮을수록 신뢰 mayby max 100
#         , top_p= 0.1#defalut 0.9 값이 높을수록 텍스트 다양 낮을수록 정확도 향상 (이게 너무 높아서 다른 주제도 끌고 온것 + 너무 낮을 경우 답변에 중요 단어가 빠짐)
#         # ex) 낮을경우 메이아이 팁 -> 메신저 알림에 대해 그냥 사용하라고만 나옴 적정 0.3
#         # 근데 일부 어느정도 조절을 해야지 비슷한 단어도 잘 찾을텐데 특히 temperature
#     )
# ollama LLM
# def getLlm(streaming_callback):
    
#     return Ollama(model=MODEL_NM
#                 , callback_manager=CallbackManager([streaming_callback])
#                 , base_url=LANGSERVE_ADD
#                 , num_ctx= 2048#defalut 2048 컨텍스트 창 설정 (질문 토큰)
#                 , num_predict= 512 #defalut 64 답변 토큰 최대 수 -1= 무한생성, -2 = 컨텐스트내에서 최대

#                 # , mirostat= 2 #defalut 0, 0 = 비활성화, 1 = Mirostat, 2 = Mirostat 2.0
#                 # , mirostat_eta= 0.5 # defalut 0.1 #응답 속도에 따른 정확도 설정 낮을 수록 천천히 대답하고 정확도 상승
#                 # , mirostat_tau= 3.0 #defalut 5.0 값이 낮을수록 텍스트가 더 집중적이고 일관성 증진 낮을 수록 다양성
#                 # mirostat 같은 경우 채팅의 일관성 등이 필요 할 때 사용

#                 # , repeat_last_n= 64#defalut 64 반복방지 n개의 토큰을 기억하고 반복 금지  128 이하
#                 , repeat_penalty= 1.3# defalut 1.1 이전 토큰 반복에 대한 처벌 높을수록 같은 토큰X 1.3미만 요망
#                 , temperature= 0.0 # 단어 출력 확률을 집중하는지,  기본값 0.8 낮을수록 신뢰 mayby max 100
#                 #, tfs_z= 1.0 #defalut 1.0 tail free Sampling <-(비활성화) 출력 가능성이 낮은 토큰 제외
#                 , top_k= 10#defalut 40 높을 수록 창의적 낮을수록 정확
#                 # ex) 낮을경우 질문에 대한 대답만 하며 높을 경우 부연 설명 추가 적정-> 40
#                 , top_p= 0.1#defalut 0.9 top-k와 작동 값이 높을수록 텍스트 다양 낮을수록 정확도 향상 (이게 너무 높아서 다른 주제도 끌고 온것 + 너무 낮을 경우 답변에 중요 단어가 빠짐)
#                 # ex) 낮을경우 메이아이 팁 -> 메신저 알림에 대해 그냥 사용하라고만 나옴 적정 0.3
#                 # 근데 일부 어느정도 조절을 해야지 비슷한 단어도 잘 찾을텐데 특히 temperature
#     )
# def getEmbLlm():
#     return OllamaEmbeddings(
#         model=EMB_MODEL_NM
#         , base_url=EMB_LANGSERVE_ADD
#         #, num_ctx= 2048#defalut 2048 컨텍스트 창 설정 (질문 토큰)
#         #, num_predict= 512 #defalut 64 답변 토큰 최대 수 -1= 무한생성, -2 = 컨텐스트내에서 최대

#         #  , mirostat= 2 #defalut 0, 0 = 비활성화, 1 = Mirostat, 2 = Mirostat 2.0
#         #  , mirostat_eta= 0.5 # defalut 0.1 #응답 속도에 따른 정확도 설정 낮을 수록 천천히 대답하고 정확도 상승
#         #  , mirostat_tau= 3.0 #defalut 5.0 값이 낮을수록 텍스트가 더 집중적이고 일관성 증진 낮을 수록 다양성
#         # mirostat 같은 경우 채팅의 일관성 등이 필요 할 때 사용

#         #, repeat_last_n= 64#defalut 64 반복방지 n개의 토큰을 기억하고 반복 금지  128 이하
#         , repeat_penalty= 1.3# defalut 1.1 이전 토큰 반복에 대한 처벌 높을수록 같은 토큰X 1.3미만 요망
#         , temperature= 0.0 # 단어 출력 확률을 집중하는지,  기본값 0.8 낮을수록 신뢰 mayby max 100
#         #, tfs_z= 2.0 #defalut 1.0 tail free Sampling <-(비활성화) 출력 가능성이 낮은 토큰 제외
#         , top_k= 80#defalut 40 높을 수록 창의적 낮을수록 정확
#         # ex) 낮을경우 질문에 대한 대답만 하며 높을 경우 부연 설명 추가 적정-> 40
#         , top_p= 0.8#defalut 0.9 top-k와 작동 값이 높을수록 텍스트 다양 낮을수록 정확도 향상 (이게 너무 높아서 다른 주제도 끌고 온것 + 너무 낮을 경우 답변에 중요 단어가 빠짐)
#         # ex) 낮을경우 메이아이 팁 -> 메신저 알림에 대해 그냥 사용하라고만 나옴 적정 0.3
#         # 근데 일부 어느정도 조절을 해야지 비슷한 단어도 잘 찾을텐데 특히 temperature
#     )

# def getAgentLlm():
#     return Ollama(
#         model=AGENT_MODEL_NM
#         , base_url=EMB_LANGSERVE_ADD
#         #, num_ctx= 2048#defalut 2048 컨텍스트 창 설정 (질문 토큰)
#         #, num_predict= 512 #defalut 64 답변 토큰 최대 수 -1= 무한생성, -2 = 컨텐스트내에서 최대

#         #, repeat_last_n= 64#defalut 64 반복방지 n개의 토큰을 기억하고 반복 금지  128 이하
#         # , repeat_penalty= 1.3# defalut 1.1 이전 토큰 반복에 대한 처벌 높을수록 같은 토큰X 1.3미만 요망
#         , temperature= 2.0 # 단어 출력 확률을 집중하는지,  기본값 0.8 낮을수록 신뢰 mayby max 100
#         #, tfs_z= 2.0 #defalut 1.0 tail free Sampling <-(비활성화) 출력 가능성이 낮은 토큰 제외
#         , top_k= 10#defalut 40 높을 수록 창의적 낮을수록 정확
#         # ex) 낮을경우 질문에 대한 대답만 하며 높을 경우 부연 설명 추가 적정-> 40
#         , top_p= 0.1#defalut 0.9 top-k와 작동 값이 높을수록 텍스트 다양 낮을수록 정확도 향상 (이게 너무 높아서 다른 주제도 끌고 온것 + 너무 낮을 경우 답변에 중요 단어가 빠짐)
#         # ex) 낮을경우 메이아이 팁 -> 메신저 알림에 대해 그냥 사용하라고만 나옴 적정 0.3
#         # 근데 일부 어느정도 조절을 해야지 비슷한 단어도 잘 찾을텐데 특히 temperature
#     )
# def getAgentAnswerLlm():
#     return Ollama(
#         model=MODEL_NM
#         , base_url=EMB_LANGSERVE_ADD
#         #, num_ctx= 2048#defalut 2048 컨텍스트 창 설정 (질문 토큰)
#         #, num_predict= 512 #defalut 64 답변 토큰 최대 수 -1= 무한생성, -2 = 컨텐스트내에서 최대

#         #, repeat_last_n= 64#defalut 64 반복방지 n개의 토큰을 기억하고 반복 금지  128 이하
#         , repeat_penalty= 2.0# defalut 1.1 이전 토큰 반복에 대한 처벌 높을수록 같은 토큰X 1.3미만 요망
#         , temperature= 0.0 # 단어 출력 확률을 집중하는지,  기본값 0.8 낮을수록 신뢰 mayby max 100
#         #, tfs_z= 2.0 #defalut 1.0 tail free Sampling <-(비활성화) 출력 가능성이 낮은 토큰 제외
#         , top_k= 10#defalut 40 높을 수록 창의적 낮을수록 정확
#         # ex) 낮을경우 질문에 대한 대답만 하며 높을 경우 부연 설명 추가 적정-> 40
#         , top_p= 0.1#defalut 0.9 top-k와 작동 값이 높을수록 텍스트 다양 낮을수록 정확도 향상 (이게 너무 높아서 다른 주제도 끌고 온것 + 너무 낮을 경우 답변에 중요 단어가 빠짐)
#         # ex) 낮을경우 메이아이 팁 -> 메신저 알림에 대해 그냥 사용하라고만 나옴 적정 0.3
#         # 근데 일부 어느정도 조절을 해야지 비슷한 단어도 잘 찾을텐데 특히 temperature
#     )
