up: ## start all services
	docker compose up -d --build
down: ## stop & remove
	docker compose down
logs: ## tail api logs
	docker compose logs -f --tail=100 api
ingest: ## run ingestion
	curl -X POST http://localhost:8000/ingest
chat: ## quick chat
	curl -s -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d '{"session_id":"dev","message":"สรุปจากเอกสารทั้งหมด"}' | jq