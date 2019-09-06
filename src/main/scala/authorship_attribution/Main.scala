package authorship_attribution

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{HashingTF, IDF, IndexToString, StopWordsRemover, StringIndexer, Tokenizer}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, SparkSession}

object Main {
  val spark = SparkSession
    .builder()
    .appName("authorship-attribution")
    .master("local")
    .getOrCreate()

  def main(args: Array[String]): Unit = {
    //get training dataset
    val csv: DataFrame = spark.read
      .option("header", "true")
      .csv(".\\src\\main\\resources\\train.csv")
    csv.printSchema()

    val Array(trainingData, testData) = csv.randomSplit(Array(0.9, 0.1), seed = 1234L)
    trainingData.cache()

    // create pipeline
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("token")

    val remover = new StopWordsRemover()
      .setInputCol("token")
      .setOutputCol("filtered")
      .setStopWords(StopWordsRemover.loadDefaultStopWords("english"))

    val tfHasher = new HashingTF()
      .setInputCol("filtered")
      .setOutputCol("rawFeatures")
      .setNumFeatures(9000)

    val idfMaker = new IDF()
      .setInputCol("rawFeatures")
      .setOutputCol("features")

    val labelIndexer = new StringIndexer()
      .setInputCol("author")
      .setOutputCol("label")
      .setHandleInvalid("keep")
      .fit(csv)

    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    val classifier = new NaiveBayes()
      .setFeaturesCol("features")
      .setLabelCol("label")

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, tfHasher, idfMaker, labelIndexer, classifier, labelConverter))

    val paramGrid = new ParamGridBuilder()
      .addGrid(tfHasher.numFeatures, Array(8000, 10000, 12000))
      .build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new MulticlassClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(5)
      .setParallelism(2)

    val model = cv.fit(trainingData)

    val res = model.transform(testData)
    res.show()

    // evaluate model
    val predictionAndLabel = res.select("prediction", "label").rdd
      .map(row => (row.get(0).asInstanceOf[Double], row.get(1).asInstanceOf[Double]))

    val metrics = new MulticlassMetrics(predictionAndLabel)
    println(metrics.weightedRecall)
    println(metrics.weightedPrecision)

    model.write.overwrite().save(".\\src\\main\\resources\\model")

    val test: DataFrame = spark.read
      .option("header", "true")
      .csv(".\\src\\main\\resources\\test.csv")

    val savedModel = CrossValidatorModel.read.load(".\\src\\main\\resources\\model")
    val testRes = savedModel.transform(test)
    testRes.show()
  }

}
