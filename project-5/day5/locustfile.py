from locust import HttpUser, between, task


class SentimentUser(HttpUser):
  wait_time = between(0.1, 0.5)

  @task
  def test_predict_endpoint(self):
    payload = {
        "text": "@United absolute worst experience ever, my luggage is missing!"
    }
    self.client.post("/predict", json=payload)