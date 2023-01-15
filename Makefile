setup:
	docker-compose up
gen:
	python producer/main.py
recon:
	python reconstructor/main.py
clean:
	docker-compose down