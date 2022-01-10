import os

BASE_DIR = os.path.dirname(__file__)

# 데이터 데이스 접속 주소
# SQLALCHEMY_DATABASE_URL = 'sqlite:///{}'.format(os.path.join(BASE_DIR, 'app.db'))
SQLALCHEMY_DATABASE_URI = 'sqlite:///{}'.format(os.path.join(BASE_DIR, 'app.db'))

# 이벤트 옵션
SQLALCHEMY_TRACK_MODIFICATIONS = False