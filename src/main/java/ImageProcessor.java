import org.apache.spark.sql.SparkSession;
import static org.apache.spark.sql.functions.*;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.api.java.UDF1;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.Image;
import java.awt.Color;
import java.io.ByteArrayInputStream;
import java.io.IOException;

public class ImageProcessor {
    public static void main(String[] args) throws IOException {
        SparkSession spark = SparkSession
            .builder()
            .appName("PneumoniaImageProcessing")
            .master("local[*]")
            .getOrCreate();

        System.out.println("Spark Session Created!");

        String hdfsPath = "hdfs://localhost:9000/images/test";

        Dataset<Row> rawImagesDF = spark.read()
            .format("binaryFile")
            .option("pathGlobFilter", "*.jpeg")
            .option("recursiveFileLookup", true)
            .load(hdfsPath);

        System.out.println("Loaded raw image data:");
        rawImagesDF.show(5);
        rawImagesDF.printSchema();
        
        Dataset<Row> labeledDF = rawImagesDF
            .withColumn("labelString", 
            regexp_extract(col("path"), ".*/(NORMAL|PNEUMONIA)/.*", 1)
        );

        System.out.println("Extracted labels:");
        labeledDF.select("path", "labelString").show(5, false);

        StringIndexer labelIndexer = new StringIndexer()
            .setInputCol("labelString")
            .setOutputCol("label")
            .setStringOrderType("alphabetAsc");

        Dataset<Row> indexedDF = labelIndexer
            .fit(labeledDF)
            .transform(labeledDF);

        System.out.println("Indexed labels:");
        indexedDF.select("labelString", "label").distinct().show();
        
        spark.udf().register(
            "preprocessUDF",
            new ImageToVectorUDF(),
            new VectorUDT()
        );

        Dataset<Row> preprocessedDF = indexedDF.withColumn(
            "features",
            callUDF("preprocessUDF", col("content"))
        );

        System.out.println("UDF applied, features column created:");
        preprocessedDF.printSchema();

        Dataset<Row> finalMLData = preprocessedDF.select("label", "features");

        System.out.println("Final DataFrame for MLlib:");
        finalMLData.show(5);

        Dataset<Row>[] splits = finalMLData.randomSplit(new double[] {0.7, 0.3}, 12345);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        System.out.println("Training data count: " + trainingData.count());
        System.out.println("Test data count: " + testData.count());

        LogisticRegression lr = new LogisticRegression()
            .setMaxIter(10)
            .setRegParam(0.01)
            .setFeaturesCol("features")
            .setLabelCol("label");

        LogisticRegressionModel model = lr.fit(trainingData);

        System.out.println("Model trained successfully!");

        // --- Evaluation on test data ---
        Dataset<Row> predictions = model.transform(testData);

        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol("label")
            .setPredictionCol("prediction")
            .setMetricName("accuracy");

        double accuracy = evaluator.evaluate(predictions);

        System.out.println("-----------------------------------------------------------------");
        System.out.println("✅ Testing Complete");
        System.out.printf("Model Accuracy on Test Data: %.4f%n", accuracy);
        System.out.println("-----------------------------------------------------------------");
        System.out.println("Sample Predictions:");
        predictions.select("label", "prediction").show(5);

        model.write().overwrite().save("hdfs://localhost:9000/models/pneumonia_classifier");
        System.out.println("Model saved to HDFS!");

        spark.stop();
    }

    private static class ImageToVectorUDF implements UDF1<byte[], Vector> {

        private static final int TARGET_WIDTH = 128;
        private static final int TARGET_HEIGHT = 128;

        @Override
        public Vector call(byte[] imageBytes) throws IOException {
            
            ByteArrayInputStream bais = new ByteArrayInputStream(imageBytes);
            BufferedImage originalImage = ImageIO.read(bais);
            bais.close();

            if (originalImage == null) {
                return Vectors.dense(new double[TARGET_WIDTH * TARGET_HEIGHT]);
            }

            BufferedImage resizedImage = new BufferedImage(
                    TARGET_WIDTH, TARGET_HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
            
            resizedImage.getGraphics().drawImage(
                    originalImage.getScaledInstance(TARGET_WIDTH, TARGET_HEIGHT, Image.SCALE_SMOOTH),
                    0, 0, null);

            double[] pixels = new double[TARGET_WIDTH * TARGET_HEIGHT];
            int i = 0;
            for (int y = 0; y < TARGET_HEIGHT; y++) {
                for (int x = 0; x < TARGET_WIDTH; x++) {
                    int gray = new Color(resizedImage.getRGB(x, y)).getRed();
                    pixels[i] = gray / 255.0;
                    i++;
                }
            }
            return Vectors.dense(pixels);
        }
    }
}
