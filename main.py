from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import stunting, mpasi, chatbot

app = FastAPI(title="Stunting")
origins = [
    "https://stuntaids.site", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(stunting.router, prefix="/stunting")
app.include_router(mpasi.router, prefix="/mpasi")
app.include_router(chatbot.router, prefix="/chatbot")
