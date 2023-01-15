setup:
	cp ./framework/templates/base.html ./producer/templates/
	cp ./framework/templates/base.html ./reconstructor/templates/
	cp ./framework/static/css/styles.css ./producer/static/css/
	cp ./framework/static/css/styles.css ./reconstructor/static/css/
	docker-compose up
gen:
	python producer/server.py
recon:
	python reconstructor/server.py
clean:
	rm ./producer/templates/base.html 
	rm ./reconstructor/templates/base.html
	rm ./producer/static/css/styles.css
	rm ./reconstructor/static/css/styles.css
	docker-compose down