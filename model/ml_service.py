import json
import os
import time

import numpy as np
import redis
import settings
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(
    host=settings.REDIS_IP, 
    port=settings.REDIS_PORT, 
    db=settings.REDIS_DB_ID,
    charset="utf-8",
    decode_responses=True
)


# Load your ML model and assign to variable `model`
model = resnet50.ResNet50(include_top=True, weights="imagenet")

def predict(image_name):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    
    # Load the image
    image_path = os.path.join(settings.UPLOAD_FOLDER, image_name)
    img = image.load_img(image_path, target_size=(224,224))

    # Resize the image to (224, 224) & transform into array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Preprocess the image for the ResNet50
    preprocessed_image = preprocess_input(img_array)

    # Make predictions
    predictions = model.predict(preprocessed_image)

    # Decode the predictions
    decoded_predictions = decode_predictions(predictions, top=1)[0]
    class_name = decoded_predictions[0][1]
    pred_probability = round(decoded_predictions[0][2],4)
    print(pred_probability)

    return class_name, pred_probability


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop the code should:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data
        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent
     
        _, msg =db.brpop(settings.REDIS_QUEUE)
        msg=json.loads(msg)
        clas, pred= predict(msg['image_name'])
        pred=str(pred)
        dict2={'prediction':clas,'score':pred}
        res_id=msg["id"]
        job_data=json.dumps(dict2)
        print(job_data)
        try:
            db.set(res_id,job_data)
        except:
            raise SystemExit ('error: result not stored')
        
        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)


def predict_batch(image_names):
    """
    Load image from the corresponding folder based on a batch of images 
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    x_batches = []

    for image_name in image_names: 

        class_name = None
        pred_probability = None

        # Preprocess the image
        # get the image from the upload folderr
        path_image = settings.UPLOAD_FOLDER + "/" + image_name
        
        # Load and preprocess the image
        img = image.load_img(path_image, target_size=(224, 224))
        # We need to convert the PIL image to a Numpy
        # array before sending it to the model
        x = image.img_to_array(img)
        # Also we must add an extra dimension to this array
        # because our model is expecting as input a batch of images.
        # In this particular case, we will have a batch with a single
        # image inside
        #x_batch = np.expand_dims(x, axis=0)
        x_batches.append(x)

    # Stack the preprocessed images into a batch
    nx_batches = np.stack(x_batches, axis=0)
    print(f"batch_image: {nx_batches.shape}")

    # Preprocess the batch
    x_batchs = preprocess_input(nx_batches)

    # Get predictions from the model
    preds = model.predict(x_batchs)
    print("number_of_predictions: ",len(preds))

    outputs = []
    for pred in preds:
        class_pred = {}
        batch_pred = np.expand_dims(np.array(pred), axis=0)
        res_model = decode_predictions(batch_pred, top=1)[0]
        class_pred["class_name"] = res_model[0][1]
        class_pred["pred_prob"] = res_model[0][2]
        outputs.append(class_pred)

    return outputs


def classify_process_batch():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes ten jobs at once from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """

    while True:
        try:
            with db.pipeline() as db_pip:
                #Take a batch of 10 new jobs from Redis with lpop
                msg_list = db.lpop(settings.REDIS_QUEUE, 10)
                if(msg_list):

                    msgs_loaded = [json.loads(msg) for msg in msg_list]
                    image_names = [json.loads(msg)["image_name"] for msg in msg_list]
                    predictions = predict_batch(image_names)

                    # Store the results in Redis
                    for msg, prediction in zip(msgs_loaded, predictions):     
                        job_data = json.dumps({
                            "prediction": prediction["class_name"],
                            "score": np.float64(prediction["pred_prob"])
                        })
                        
                        db_pip.set(msg["id"], job_data)
                    db_pip.execute()
                    
                    # Sleep for a bit inside of batch
                    time.sleep(settings.SERVER_SLEEP)
        except Exception as exc:
            print(exc)



if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()

