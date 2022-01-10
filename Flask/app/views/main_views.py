from flask import Blueprint, render_template
from app.models import Question


bp = Blueprint('main', __name__, url_prefix='/bp')

@bp.route('/hello')
def hello():
    return 'hello! 3'


@bp.route('/')
def index():
    question_list = Question.query.order_by(Question.question_create_time.desc())
    

    return 'index'