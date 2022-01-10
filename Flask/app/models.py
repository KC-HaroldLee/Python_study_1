from app import db

class Question(db.Model) :
    question_id = db.Column(db.Integer, primary_key=True)
    question_subject = db.Column(db.String(30), nullable=False)
    question_content = db.Column(db.Text, nullable=False)
    question_create_time = db.Column(db.DateTime(), nullable=False)


class Answer(db.Model) :
    answer_id = db.Column(db.Integer, primary_key=True)
    question_id = db.Column(db.Integer, db.ForeignKey('question.question_id', ondelete='CASCADE'))
    answer_question = db.relationship('Question', backref=db.backref('answer_set', ))
    answer_content = db.Column(db.Text, nullable=False)
    answer_create_time = db.Column(db.DateTime(), nullable=False)

