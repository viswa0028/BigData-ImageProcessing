# BigData Preprocessing

This project loads chest X-ray images, preprocesses them into fixed-size grayscale feature vectors, trains a Logistic Regression classifier using Apache Spark MLlib, evaluates the model, and saves the trained model to disk.

## Repository structure

- `pom.xml` - Maven project file with Spark dependencies
- `src/main/java/ImageProcessor.java` - Main application: loads images, preprocesses, trains, evaluates, saves model
- `target/` - Maven build output
- `images/test/` - (Expected) local folder with images organized as `.../NORMAL/*.jpeg` and `.../PNEUMONIA/*.jpeg`

## Prerequisites

- macOS (or Linux)
- Java 8+ (JDK)
- Maven
- (Optional) Hadoop/HDFS if you want to read/write from `hdfs://` URIs

Verify Java and Maven:

```bash
java -version
mvn -version
```

## How the code works (high-level)

1. Creates a SparkSession in `local[*]` mode for local testing.
2. Reads images using Spark's `binaryFile` data source (supports recursive lookup).
3. Extracts labels from the file path using a regex (`NORMAL` or `PNEUMONIA`).
4. Registers a UDF that converts image bytes to a 128×128 grayscale flattened `Vector` (normalized to [0,1]).
5. Builds a DataFrame with `label` and `features` columns and splits it into training/test sets (70/30).
6. Trains a `LogisticRegression` model and evaluates accuracy using `MulticlassClassificationEvaluator`.
7. Saves the trained model to a local path (by default) or HDFS.

## Build

From the project root:

```bash
mvn clean package -DskipTests
```

## Run (local images + local model save)

Make sure your images are placed under a local folder such as:

```
/Users/<your-username>/path/to/project/images/test/NORMAL/*.jpeg
/Users/<your-username>/path/to/project/images/test/PNEUMONIA/*.jpeg
```

Then run the application (zsh):

```bash
# Add the opens flags required for Spark + Java module access
java \
  --add-opens=java.base/sun.nio.ch=ALL-UNNAMED \
  --add-opens=java.base/java.io=ALL-UNNAMED \
  -cp "target/spark-project-1.0.jar:$(mvn -q dependency:build-classpath -Dmdep.outputFile=/dev/stdout 2>/dev/null)" \
  ImageProcessor
```

Notes:
- The application defaults to a local image path (e.g., `/Users/viswa/Desktop/data/FInal Project/images/test`). Edit `ImageProcessor.java` to change `imagePath` or to use HDFS URIs.
- The model is saved by default to `models/pneumonia_classifier` under the project root. Change `modelPath` inside the code to modify.

## Run (using HDFS)

If you prefer HDFS, ensure your Hadoop services (NameNode + DataNode) are running and the HDFS URI in `ImageProcessor.java` points to the correct namenode, e.g. `hdfs://localhost:9000/images/test`.

Start HDFS (if installed):

```bash
start-dfs.sh
start-yarn.sh
```

If you get a `Connection refused` error when the app attempts to read `hdfs://...`, the namenode is not reachable. See Troubleshooting below.

## Example output

When successful you will see steps printed like:

- Spark Session Created!
- Loaded raw image data:
- Extracted labels
- UDF applied, features created
- Training data count / Test data count
- Model trained successfully!
- Testing Complete
- Model Accuracy on Test Data: 0.7757
- Sample Predictions:

And the model saved to the configured path.

## Troubleshooting

- SLF4J warnings (``SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder"``): harmless; they mean a logging backend isn't on the classpath. The app still runs.

- `Connection refused` to `hdfs://localhost:9000`: means HDFS is not running or the namenode host/port is wrong. Either start HDFS or change to a local path in the code.

- `IllegalAccessError` related to `sun.nio.ch.DirectBuffer`: run Java with the `--add-opens` flags shown in the Run section.

- Compilation errors after editing sources: run `mvn -q clean package -DskipTests` and check the `target/` folder for compiled classes.

## Configuration tweaks useful for development

- Change image size: inside `ImageToVectorUDF` modify `TARGET_WIDTH` / `TARGET_HEIGHT`.
- Change train/test split: modify the randomSplit weights in the main method.
- Add more evaluation metrics: use `MulticlassClassificationEvaluator` with `precisionByLabel`, `recallByLabel`, or compute confusion matrix with `MulticlassMetrics`.

## Next steps / enhancements

- Try convolutional models (TensorFlow / PyTorch) for better image performance.
- Add cross-validation and hyperparameter tuning using `CrossValidator` / `ParamGridBuilder` from Spark ML.
- Store predictions and evaluation metrics to a CSV or a monitoring store.

## Run the code using
```bash
cd "/Users/viswa/Desktop/data/FInal Project" && mvn -q clean package -DskipTests
```
```bash
cd "/Users/viswa/Desktop/data/FInal Project" && java --add-opens=java.base/sun.nio.ch=ALL-UNNAMED --add-opens=java.base/java.io=ALL-UNNAMED -cp "target/spark-project-1.0.jar:$(mvn -q dependency:build-classpath -Dmdep.outputFile=/dev/stdout 2>/dev/null)" ImageProcessor 2>&1
```

