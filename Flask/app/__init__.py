from flask import Flask

# 처음 테스트
# @app.route('/')
# def hello():
#     return 'hello!'


# app.py에서 __init__.py로 변경
# def create_app() :
#     app = Flask(__name__) 
#     @app.route('/')
#     def hello():
#         return 'hello! 2'
#     return app


# 블루프린트 적용
# def create_app() :
#     app = Flask(__name__) 

#     from .views import main_views
#     app.register_blueprint(main_views.bp)

#     return app

from flask_migrate import Migrate, migrate
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy() 
migrate = Migrate()

import config # 내가 만든 config.py

def create_app() :
    app = Flask(__name__) 

    app.config.from_object(config) # 내가 만든 config.py

    # ORM 
    db.init_app(app)
    migrate.init_app(app, db)
    from . import models

    # 블루프린트
    from .views import main_views
    app.register_blueprint(main_views.bp)

    return app