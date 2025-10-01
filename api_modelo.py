import os
import logging
import datetime
import jwt
from functools import wraps
from flask import Flask, request, jsonify
import joblib
import numpy as np
from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

JWT_SECRET = "secreto"
JWT_ALGORITHM = "HS256"
JWT_EXP_DELTA_SECONDS = 3600

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api_modelo")

db_url = "sqlite:////tmp/predictions.db"
engine = create_engine(db_url, echo=False)
Base = declarative_base()
SessionLocal = sessionmaker(bind=engine)

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sepal_length = Column(Float, nullable=False)
    sepal_width = Column(Float, nullable=False)
    petal_length = Column(Float, nullable=False)
    petal_width = Column(Float, nullable=False)
    predicted_class = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(engine)

model = joblib.load("modelo_iris.pkl")
logger.info("Modelo carregado com sucesso.")

#Iniciar aplicação Flask
app = Flask(__name__)
predictions_cache = {}

# Autenticação JWT
test_username = "admin"
test_password = "secret"

def create_token(username):
    # O 'payload' são as informações que queremos armazenar dentro do token.
    payload = {
        "username": username,
        # 'exp' (expiration) é uma claim padrão do JWT que define o tempo de vida do token.
        # Aqui, estamos a definir que o token expira no momento atual mais um determinado número de segundos.
        "exp": datetime.datetime.utcnow() + datetime.timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    # jwt.encode cria o token, combinando o payload com uma chave secreta e um algoritmo de criptografia.
    # A chave secreta nunca deve ser exposta publicamente.
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def token_required(f):
    # @wraps(f) é um decorator que ajuda a preservar a identidade da função original (como o seu nome e docstring).
    @wraps(f)
    def decorated(*args, **kwargs):
        # Esta função interna é a que será executada no lugar da função original protegida.
        
        # Lógica para pegar o token do header Authorization: Bearer <token>
        # Lógica para descodificar o token, checar a assinatura e a data de expiração
        return f(*args, **kwargs)
    
    # O decorator 'token_required' retorna a nova função 'decorated'.
    return decorated

@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    username = data.get("username")
    password = data.get("password")
    if username == test_username and password == test_password:
        token = create_token(username)
        return jsonify({"token": token})
    else:
        return jsonify({"error": "Credenciais inválidas"}), 401

@app.route("/predict", methods=["POST"])
@token_required
def predict():
    """
    Endpoint protegido por token para obter predição.
    Corpo (JSON):
    {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }
    """
    # Pega os dados JSON enviados no corpo da requisição. `force=True` ignora o cabeçalho content-type.
    data = request.get_json(force=True)
    # Bloco try-except para garantir que os dados de entrada são válidos.
    try:
        # Converte cada valor do JSON para o tipo float (número com casas decimais).
        sepal_length = float(data["sepal_length"])
        sepal_width = float(data["sepal_width"])
        petal_length = float(data["petal_length"])
        petal_width = float(data["petal_width"])
    # Captura erros se uma chave não existir (KeyError) ou se o valor não puder ser convertido para float (ValueError).
    except (ValueError, KeyError) as e:
        # Regista o erro para depuração.
        logger.error("Dados de entrada inválidos: %s", e)
        # Retorna uma mensagem de erro clara para o utilizador.
        return jsonify({"error": "Dados inválidos, verifique parâmetros"}), 400

    # Cria uma tupla com as características para usar como chave no cache.
    features = (sepal_length, sepal_width, petal_length, petal_width)
    # Verifica se a predição para estas características já existe no cache.
    if features in predictions_cache:
        # Regista que a predição foi encontrada no cache (cache hit).
        logger.info("Cache hit para %s", features)
        # Pega o resultado diretamente do cache, evitando reprocessamento.
        predicted_class = predictions_cache[features]
        source = "cache"
    else:
        # Se não estiver no cache, executa o modelo de machine learning.
        # Prepara os dados de entrada no formato que o modelo espera (geralmente um array NumPy).
        input_data = np.array([features])
        # Usa o modelo carregado para fazer a predição.
        prediction = model.predict(input_data)
        # Extrai o resultado da predição e converte-o para um inteiro.
        predicted_class = int(prediction[0])
        # Armazena o novo resultado no cache para acelerar futuras requisições com os mesmos dados.
        predictions_cache[features] = predicted_class
        # Regista que o cache foi atualizado com a nova predição.
        logger.info("Cache updated para %s", features) 
        source = "model"
    
    # Salva a predição no banco de dados
    db = SessionLocal()
    new_prediction = Prediction(
        sepal_length=sepal_length,
        sepal_width=sepal_width,
        petal_length=petal_length,
        petal_width=petal_width,
        predicted_class=predicted_class
    )
    db.add(new_prediction)
    db.commit()
    db.close()

    # Retorna a predição e a sua origem (cache ou modelo).
    return jsonify({
        "predicted_class": predicted_class,
        "source": source
    })        

@app.route("/predictions", methods=["GET"])
@token_required
def list_predictions():
    """
    Lista as predições armazenadas no banco.
    Parâmetros opcionais (via query string):
        - limit (int): quantos registros retornar, padrão 10
        - offset (int): a partir de qual registro começar, padrão 0
    Exemplo:
    /predictions?limit=5&offset=10
    """
    limit = int(request.args.get("limit", 10))
    offset = int(request.args.get("offset", 0))
    db = SessionLocal()
    preds = db.query(Prediction).order_by(Prediction.id.desc()).limit(limit).offset(offset).all()
    db.close()
    results = []
    for p in preds:
        results.append({
            "id": p.id,
            "sepal_length": p.sepal_length,
            "sepal_width": p.sepal_width,
            "petal_length": p.petal_length,
            "petal_width": p.petal_width,
            "predicted_class": p.predicted_class,
            "created_at": p.created_at.isoformat()
        })
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True, port=1312)