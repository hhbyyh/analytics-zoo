## Object Detection example
This example illustrates how to detect objects in image with pre-trained model.

### Run steps
1. Install OpenCV
The example uses OpenCV library to save image. Please install it before run this example.

2. Prepare pre-trained models

Download pre-trained models from https://github.com/intel-analytics/zoo/tree/master/docs/models/objectdetection

3. Prepare predict dataset

Put your image data for prediction in the ./image folder.

4. Run the example

```bash
modelPath=... // model path. Local file system/HDFS/Amazon S3 are supported

imagePath=... // image path. Local file system/HDFS are supported. With local file system, the files need to be available on all nodes in the cluster.

outputPath=... // output path. Currently only support local file system.

PYTHON_API_ZIP_PATH=${ZOO_HOME}/dist/lib/zoo-VERSION-SNAPSHOT-python-api.zip
ZOO_JAR_PATH=${ZOO_HOME}/dist/lib/zoo-VERSION-SNAPSHOT-jar-with-dependencies.jar
PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH

spark-submit \
    --master local[4] \
    --driver-memory 10g \
    --executor-memory 10g \
    --py-files ${PYTHON_API_ZIP_PATH} \
    --jars ${ZOO_JAR_PATH} \
    --conf spark.driver.extraClassPath=${ZOO_JAR_PATH} \
    --conf spark.executor.extraClassPath=${ZOO_JAR_PATH} \
    ${ZOO_HOME}/pyzoo/zoo/examples/objectdetection/predict.py ${model_path} ${image_path} ${output_path}
```

## Results
You can find new generated images stored in output_path, and the objects in the images are with a box around them [labeled "name"]