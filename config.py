from typing import Dict
conf: Dict[str, str] = {
        'os': 'window',
        #'os' : 'linux',

        #임베딩 최소 문서 제한
        'docCount' : 5,

        #Model 명
        'modelNm' : 'bnksys/yanolja-eeve-korean-instruct-10.8b',
        # 'modelNm' : 'llava:13b',
        # 'modelNm' : 'llama3.1:lastest',
        
        'assiModelNm' : 'bnksys/yanolja-eeve-korean-instruct-10.8b',

        # 'embModelNm' : 'paraphrase-multilingual', # 말 길게하는데 핵심 빼먹거나 환각의 경우가 간혹 있는듯
        'embModelNm' : 'chatfire/bge-m3:q8_0',    # 요약할 떄 그나마 맞는듯 환각 유의

        # 'agentModelNm' : 'bnksys/yanolja-eeve-korean-instruct-10.8b',

        #모델 서버
        'langserveAdd' : 'http://localhost:11434',
        
        'openAiLangserveAdd' : 'http://localhost:11434/v1',

        'embLangserveAdd' : 'http://localhost:11434',

        'openAiEmbeddingLangserveAdd' : 'http://localhost:11434/api'
    }