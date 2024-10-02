from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

from model.model_setup import getEmbLlm

from langchain_chroma import Chroma
# from langchain_community.vectorstores import Chroma,FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings


###########################################################
            #주의 사항#

        # QA_CHAIN_PROMPT = promptInit.BasePrompt() #=> PromptTemplate 사용 시 RetrievalQA.from_chain_type 사용
      
        #chain 생성
        # qa_chain = RetrievalQA.from_chain_type(
        #     llm,
        #     retriever=retriever3,
        #     chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        # )

        #FEW_SHOT_CHAIN_PROMPT = promptInit.FewShotPrompt()#=> ChatPromptTemplate 사용 시 FEW_SHOT_CHAIN_PROMPT | llm | retriever3 사용
        #qa_chain = FEW_SHOT_CHAIN_PROMPT | llm | retriever3
############################################################## 

#일반 CHAT 사용시 프롬프트 템플릿
CHAT_PROMPT_TEMPLATE = """You are an analytical AI. Answer the questions concisely. If you don't know the answer, say you don't know.  <Question>: {input}"""
#RAG 사용시 프롬프트 템플릿
RAG_PROMPT_TEMPLATE = """당신은 내용 정리 AI이다. 많은 주제에 대한 개별 내용을 요약해야한다. 
                        입력된 문맥들을 짧고 간단하게 요약하여 답해. 
                        비용과 금액에 대한 문서는 제외해. 
                        단순 텍스트 형식으로 정보를 제공해.
Question: {question} 
Context: {context} 
Answer:"""

# 보조 chain 반복 실행 시 prompt 생성 
ASSI_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""당신은 내용 정리 AI입니다. 다음 요구 사항을 충족하여 답변을 작성하세요:
1. 이름, 주소, 연락처 등 개인을 식별할 수 있는 정보는 제외하세요.
2. 금액, 날짜, 연도 등 구체적인 숫자 정보는 제외하세요.
3. 답변이 모호하거나 잘 모르겠다면 '모른다'고 명확히 답변하세요.

Question: {question}
Context: {context}
Answer:"""
)
ASSI_ITREATOR_PROMPT_TEMPLATE ="당신은 분석 AI이다. 많은 주제에 대한 개별 내용을 요약해야한다. 반복 되는 글자를 기준으로 분류 하여 요약해라. 입력된 문맥들을 짧고 간단하게 요약하여 답해. 단순 텍스트 형식으로 정보를 제공해야 한다."

# 내용 요약 답변 프롬프트
# LAST_PROMPT_TEMPLATE = """당신은 내용 정리 AI이다. 제공 된 문서 및 메모리를 통해 얻은 내용을 제공해야한다. 단순 텍스트 형식으로 정보를 제공해야한다.
LAST_PROMPT_TEMPLATE = """You are a content organizing AI. Information obtained through provided documentation and memory must be provided. Information must be provided in simple text format
Question: {question} 
Context: {context} 
Answer:"""

############################################################################
                # crew ai 사용시 프롬프트
#어시 요청
ASSI_DESCRIPTION = """제목을 통한 분류를 하며 Tool의 내용을 설명. """
#목표
AGENT_GOAL = """Tool의 요청 정보를 요약 해야한다."""
#어시 목표
ASSI_GOAL = """Tool을 사용하여 나온 데이터를 분류"""

#역할 부여
AGENT_BACKSTORY = """당신은 요약 AI 입니다. 조수가 정리 한 내용을 토대로 자료를 분류하여 답변. 입력된 문맥들을 짧고 짧고 간단하게 요약"""
#어시 역할 부여
ASSI_BACKSTORY ="""당신은 조수 AI이다. Tool 정보 문맥들의 핵심주제를 정리해라. """

#출력 내용 설명
AGENT_EXPECTED_OUTPUT = """제시 된 Tool 내용 정리. 짧고 단순 텍스트 형식으로 정보를 제공"""
#어시 출력 내용 설명
ASSI_TASK_EXPECTED_OUTPUT = """단순 텍스트 형식으로 정보를 제공해. 요약 시 개인정보(이름, 서명, 주소, 전화번호 등)는 포함하면 안돼."""
############################################################################


# EXAMPLES = [
#             {"input": "메이아이 설명서", "output": """'메이아이 설명서'의 메일 작업 관리, 연차관리, 비용 관리, 차량 이용 규정, 근로제도, 입사 후 제출 서류 의 주요 내용입니다"""},
#             {"input": """총장임용후보자 선정에 관한 규정 및 시행세칙 일부개정안 의견 수렴"""
#              , "output": """'총장임용후보자 선정에 관한 규정 및 시행세칙 일부개정안'의 선거권 및 투표수 관련 조항 수정, 피선거권에 관한 규정 업데이트, 선거 후보자 등록 관련 조항 변경, 선거벽보 관련 조항 수정 주요 내용입니다"""},
#             {"input": """공무직(조리원) 제한경쟁특별채용시험 공고"""
#              , "output": """'채용시험 공고'의 모집 분야 및 인원, 응시 자격 요건, 시험 일정 및 장소, 합격 기준, 면접시험 세부사항 및 합격자 발표 주요 내용입니다 """},
#             {"input": """평의회 회의록 내용 요약"""
#              , "output": """'평의회 회의록'의 총장 직무대행 인사말, 대학평의원회의 결과 보고, 평가 준비사항 점검 및 향후 계획 수립, 기타 안건으로 학사제도 개선 주요 내용입니다"""},
#             {"input": """[지방보조금]인성교육 지원 사업 선정에 따른 프로그램 수강 신청 안내"""
#              , "output": """'인성 교육 지원에 관한 「지방 보조금」 공고 및 관련 내용' 목적, 대상 기업 선정 기준, 프로그램 내용 및 운영 방식, 신청 방법, 선정 절차, 기타 사항 및 결과 발표 요약입니다"""},
# ]
EXAMPLES = [
#            {"input": "메이아이 설명서", 
#             "output": """'메이아이 설명서'의 주요 내용입니다:
#                    ** 메일 작업 관리: 부서별 주간 및 월간 보고서 조회와 등록을 할 수 있습니다.
#                    ** 연차관리: 직원들은 유급 휴가인 '연차', 반일 근무를 위한 '반차' 및 기타 사유(경조사, 병가 등)에 대한 시간을 등록할 수 있습니다.
#                    ** 비용 관리: 급여와 관련된 다양한 비용을 추적하고 분류합니다
#                    ** 차량 이용 규정: 회사차량 사용에 대한 지침과 우선순위를 명시하며 차종별 배정 정보를 제공합니다.
#                    ** 입사 후 제출 서류: 신규 직원에게 필요한 문서들을 나열하고 있습니다.
#                    ** 근로제도: 근무시간(09:00~18:00), 회의 및 기타 관련 정보를 제공합니다."""},
            {"input": """조리원 채용 시험 안내문 요약"""
             , "output": """'조리원 채용 시험 안내문'의 주요 내용입니다: 
                    ** 모집 분야 및 인원 : 총 두 명의 정규조리원을 선발
                    ** 응시 자격 요건 : 조리기능사 또는 식품위생관리사(또는 이와 동등자격증 소지자), 교육 이수 증빙서류 제출 가능
                    ** 시험 일정 및 장소 : 필기시험이 실시되며 면접심사는 추후 통보 예정입니다.
                    ** 합격 기준 : 각 과목을 별도로 계산하여 평균 점수가 각각 만점의 절반 이상을 취득한 자를 선발합니다.
                    ** 면접시험 세부사항 및 합격자 발표 : 필기 시험 합격자에 한해 진행되며 최종합격자는 홈페이지에 게시됩니다. """},
            {"input": """총장임용후보자 선정에 관한 규정 및 시행세칙 일부개정안 요약"""
             , "output": """'총장임용후보자 선정에 관한 규정 및 시행세칙 일부개정안'의 주요 내용입니다:
                    ** 선거권 및 투표수 관련 조항 수정 : 교수, 직원(교수를 포함), 학생에 대한 투표 비율을 변경하였습니다 .
                    ** 피선거권에 관한 규정 업데이트 : 선거나 공고일 현재 본교 교수는 전·현직 총장 이력자가 아닌 교수 또는 부교수로 한정되었습니다.
                    ** 선거 후보자 등록 관련 조항 변경 : 다음 각 호에 해당하는 직을 가진 사람이 입후보하려면 그 시기는 공고일로부터 현 총장 임기만료까지로 조정되어야 합니다.
                    ** 선거벽보 관련 조항 수정 : 추천위원회에 공고일 전날 제출해야 한다는 내용이 추가되었습니다"""},
            {"input": """평의회 회의록 내용 요약"""
             , "output": """'평의회 회의록'의 주요 내용입니다: 
                    ** 총장 직무대행 인사말 : 평가 준비, 교직원 복지 향상에 대한 필요성을 강조하였습니다. 
                    ** 대학평의원회의 결과 보고 : 위원회를 구성하여 관련 사항을 심의하기로 결정했습니다.
                    ** 평가 준비사항 점검 및 향후 계획 수립: : 교직원 역량 강화 프로그램 실시(온라인 강의 제작 워크숍 등)
                    """},
             {"input" : """주유비지원규정 요약"""
              ,"output" : """주유비지원규정 내용 요약입니다.
                    ** 주유비용 지원규정 : 회사 이 규정은 업무로 개인차량을 이용하는 직원에게 거리에 따른 비용을 계산하고 지급하는데 목적이 있다.
                    ** 회사차량 운영규정 : 본 규정은 회사에 보유한 회사차량에 대하여 효율적 운영과 안전한 운행을 위하여 규칙을 정의함에 있다.
                            """
             },
            {"input": """'인성교육 지원 사업 선정에 따른 프로그램 수강 신청' 요약"""
             , "output": """'인성 교육 지원에 관한 「지방 보조금」 공고 및 관련 내용' 요약입니다: 
                    ** 목적 : 사람들을 대상으로 한 인성과 창의력 함양, 인성교육 프로그램 지원. """},
            {"input" : """주간 보고서 내용 요약"""
             ,"output" : """데이터가 부족하여 데이터를 요약할 수 없습니다."""},

#예시 소수 추가 및 없는 내용시 추가 
]
# EXAMPLES = [ 
#             {"input": "메이아이 설명서", "output": """'메이아이 설명서'의 주요 내용입니다:
# 1) 메일 작업 관리: 부서별 주간 및 월간 보고서 조회와 등록을 할 수 있습니다.
# 2) 연차관리: 직원들은 유급 휴가인 '연차', 반일 근무를 위한 '반차' 및 기타 사유(경조사, 병가 등)에 대한 시간을 등록할 수 있습니다.
# 3) 비용 관리: 급여와 관련된 다양한 비용을 추적하고 분류합니다
# 4) 차량 이용 규정: 회사차량 사용에 대한 지침과 우선순위를 명시하며 차종별 배정 정보를 제공합니다.
# 5) 입사 후 제출 서류: 신규 직원에게 필요한 문서들을 나열하고 있습니다.
# 6) 근로제도: 근무시간(09:00~18:00), 회의 및 기타 관련 정보를 제공합니다."""},
#             {"input": """총장임용후보자 선정에 관한 규정 및 시행세칙 일부개정안 의견 수렴"""
#              , "output": """'총장임용후보자 선정에 관한 규정 및 시행세칙 일부개정안'의 주요 내용입니다:
# 1) 선거권 및 투표수 관련 조항 수정 : 교수, 직원(교수를 포함), 학생에 대한 투표 비율을 변경하였습니다 .
# 2) 피선거권에 관한 규정 업데이트 : 선거나 공고일 현재 본교 교직원인 교수는 전·현직 총장 이력자가 아닌 교수 또는 부교수로 한정되었습니다.
# 3) 선거 후보자 등록 관련 조항 변경 : 다음 각 호에 해당하는 직을 가진 사람이 입후보하려면 해당 직에서 사임해야 하며, 그 시기는 공고일로부터 현 총장 임기만료까지로 조정되거나 재선거 또는 궐위 선거나 경우와 같이 특정 기간 내에 이루어져야 합니다.
# 4) 선거벽보 관련 조항 수정 : 후보자는 이제 한 장의 벽보를 작성할 수 있으며, 추천위원회에 공고일 전날 제출해야 한다는 내용이 추가되었습니다"""},
#             {"input": """공무직(조리원) 제한경쟁특별채용시험 공고"""
#              , "output": """'조리원 채용 시험 안내문'의 주요 내용입니다: 
# 1) 모집 분야 및 인원 - 총 두 명의 정규조리원을 선발하며, 근무지는 예산군 덕산면 소재 .
# 2) 응시 자격 요건 - ① 조리기능사 또는 식품위생관리사(또는 이와 동등자격증 소지자), 그리고 해당 분야 경력이나 교육 이수 증빙서류 제출 가능, ② 만18세 이상이며 학교 졸업 이상의 학력을 가진 자.
# 3) 시험 일정 및 장소 - 필기시험이 실시되며 면접심사는 추후 통보 예정입니다.
# 4) 필기 시험 과목 및 합격 기준 - 조리실무와 식품위생에 관한 객관식 2과목 총 60문항을 실시하며, 각 과목을 별도로 계산하여 평균 점수가 각각 만점의 절반 이상을 취득한 자를 선발합니다.
# 5) 면접시험 세부사항 및 합격자 발표 - 필기 시험 합격자에 한해 진행되며 최종합격자는 홈페이지에 게시됩니다. """},
#             {"input": """공무직(조리원) 제한경쟁특별채용시험 공고"""
#              , "output": """'평의회 회의록'의 주요 내용입니다: 
# 1) 총장 직무대행 인사말 - 평가 준비, 교직원 복지 향상에 대한 필요성을 강조하였습니다. 
# 2) 대학평의원회의 결과 보고 - 학사제도 개선안과 과정 개편에 관한 논의가 있었으며, 위원회를 구성하여 관련 사항을 심의하기로 결정했습니다.
# 3) 평가 준비사항 점검 및 향후 계획 수립: - 교직원 역량 강화 프로그램 실시(온라인 강의 제작 워크숍 등)
# 4) 대학평의원회 위원 선임에 관한 논의. 
# 5) 기타 안건으로 '대학평가 준비 및 평가 대응'과 관련하여 교직원 역량 강화 프로그램 실시와 학사제도 개선을 통한 만족도 향상에 대한 필요성을 강조하였습니다. """},
#             {"input": """[지방보조금]인성교육 지원 사업 선정에 따른 프로그램 수강 신청 안내"""
#              , "output": """'인성 교육 지원에 관한 「지방 보조금」 공고 및 관련 내용' 요약입니다: 
# 1) 목적 - 사람들을 대상으로 한 인성과 창의력 함양, 인성교육 프로그램 지원.
# 2) 대상 기업 선정 기준 - 도내 소재 중소 규모의 기업을 우선 고려하며 교육청 추천을 받습니다
# 3) 사업 기간과 예산 범위 : 4월부터 시작하여 총사업비 약 천만 원(예산은 충남도에서 제공).
# 4) 프로그램 내용 및 운영 방식 - 인성교육 관련 다양한 활동을 진행하며, 교육청이 추천한 전문 기관을 통해 교육을 실시합니다.
# 5) 신청 방법 : 담당 교사가 3월까지 인근의 지방 보조금 사무소에 지원서를 제출해야 합니다
# 6) 선정 절차 및 결과 발표 - 심사위원회의 평가를 거쳐 최종적으로 프로그램이 승인되면, 교육청에 통보됩니다.
# 7) 기타 사항 : 사업 기간 중 발생하는 모든 비용은 학교가 부담하며, 관련 규정이나 지침에 따라 진행되어야 한다는 점을 유의하세요.' """},
# ]

#학교,학생, 교육, 평가, 공무, 평가
def FewShotPrompt(): 
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"), 
            ("ai", "{output}")
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt = example_prompt, 
        examples = EXAMPLES,
    )
    
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a content organizing AI. Each content must be explained. If there is insufficient data, you must answer that it is insufficient. Information should be provided in simple text format."),  
            few_shot_prompt,
            ("human", "{input}")            
        ]
    )
    return final_prompt

def TestFewShotPrompt(): 
    # Few Shot Prompt Selector <= 현재 OpenAiEmbedding만 호환하는 문제로 인해 사용불가
    example_prompt = ChatPromptTemplate.from_messages(
        [
            # ("human", "{context}"),# from_chain_type
            ("ai", "{output}"),
            ("human", "{input}"), # create_retrieval_chain
            # ("assistant", "{context}") 
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt = example_prompt, 
        examples = EXAMPLES,

    )
    
    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 내용 정리 AI이다. 각 내용을 설명해야 한다. 데이터가 부족할 시 부족하다고 답변해야 한다. 단순 텍스트 형식으로 정보를 제공해야 한다."),  
            few_shot_prompt,
            ("human", "{input}"),
            # ("ai", "{context}")  # Adding context to the prompt    
        ]
    )
    return final_prompt

def ChatPrompt() :
    return ChatPromptTemplate.from_template(CHAT_PROMPT_TEMPLATE)

def MemoryPrompt() :
    return PromptTemplate(
        input_variables=["context", "question"],
        template=LAST_PROMPT_TEMPLATE,
    )

def BasePrompt() :
    return PromptTemplate(
        input_variables=["context", "question"],
        template=RAG_PROMPT_TEMPLATE,
    )
#

