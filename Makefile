.PHONY: test up down health logs eval-smoke eval-answer

test:
	python -m unittest discover -s tests -t . -v

up:
	docker compose up --build -d

down:
	docker compose down

health:
	curl -sS http://127.0.0.1:8000/health

logs:
	docker compose logs --tail=100 backend

eval-smoke:
	python eval/run_retrieval_eval.py --suite smoke

eval-answer:
	python eval/run_eval.py
