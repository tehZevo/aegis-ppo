FROM stablebaselines/rl-baselines-zoo-cpu

WORKDIR /app

RUN mkdir -p /app/models

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

EXPOSE 80
VOLUME ["/app/models"]

CMD [ "python", "-u", "main.py" ]
