from locust import HttpUser, between, task


class APIUser(HttpUser):
    wait_time = between(1, 5)

    # Put your stress tests here.
    # See https://docs.locust.io/en/stable/writing-a-locustfile.html for help.

    @task
    def index(self):
        # Send an HTTP GET request
        self.client.get("/")
        self.client.get("/feedback")

    @task
    def stress_predict(self):
        data_in_files = {"file": open("dog.jpeg", "rb")}
        self.client.post(
            "/predict",
            files=data_in_files
        )
    
        
