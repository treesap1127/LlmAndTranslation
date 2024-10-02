from typing import Any, AsyncGenerator
from langchain_core.outputs import LLMResult
from langchain.callbacks.base import BaseCallbackHandler
from fastapi import WebSocket
import asyncio

task_store = {}

class StreamingCallback(BaseCallbackHandler):
    #초기화
    def __init__(self, task_id: str, websocket: WebSocket):
        self.task_id = task_id
        self.websocket = websocket
        task_store[task_id] = {
            "tokens": [],
            "streamEnd": False
        }
    # llm 토큰 생성 시 실행
    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        task_store[self.task_id]["tokens"].append(token)
#        print(self.task_id + ' 모델 테스트 ' + token)
    # llm 멘트 종료 시 실행
    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        task_store[self.task_id]["streamEnd"] = True
        await self.websocket.close()
    # 반환 token 함수
    async def token_generator(self) -> AsyncGenerator[str, None]:
        while not task_store[self.task_id]["streamEnd"]:
            if task_store[self.task_id]["tokens"]:
                token = task_store[self.task_id]["tokens"].pop(0)
                yield f"{token}"#\n\n 기준으로 응답 발송
                await asyncio.sleep(0) # yield 이벤트 우선순위 1순위로 변경 (제외 시 문장 형식으로 반환)
            else:
                await asyncio.sleep(0.1) 