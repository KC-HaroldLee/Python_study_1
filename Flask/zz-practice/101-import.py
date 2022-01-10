from flask import Flask

app = Flask(__name__)
# 플라스크 애플리케이션을 생성하는 코드.
# 이 코드에서 __name__이라는 변수에는 모듈명이 담긴다. (이 경우 '101.import')

@app.route('/')
def hello():
    return 'hello!'


