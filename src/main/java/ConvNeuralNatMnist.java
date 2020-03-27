import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class ConvNeuralNatMnist {

  // Gestion de la journalisation
    private static Logger logger= LoggerFactory.getLogger(ConvNeuralNatMnist.class);

    public static void main(String[] args) throws IOException {

      // Chemin d'acces du stockage du jeu de données mnist
        String basePath=System.getProperty("user.home")+"/mnist_png";
        // Affichage du chemin
        System.out.println(basePath);



      System.out.println("---------------------Creation des hyperparametres du reseau ----------------------------------------------------------");
        // Hauteur et Largeur des images mnist
        int height=28;int width=28;
        // 1 seul canneau donc gris
        int channels=1;// signe channel for graysacle image
      // nombre de sortie 10
        int outputNum=10;// 10 digits classification
      // Taille des echantillons
        int batchSize=54;
        // Compteur du nombre d'iterations
        int epochCount=1;
        int seed =1234;
        // Vecteur d'un emsemble de taux d'apprentissage choisis sur un ensemble d'itérations
        Map<Integer,Double> learningRateByIterations=new HashMap<>();
        // Iteration 0 on choisie un learning rate 0.06
        learningRateByIterations.put(0,0.06);
      // Iteration 200 on choisie un learning rate 0.05
        learningRateByIterations.put(200,0.05);
      // Iteration 600 on choisie un learning rate 0.028
        learningRateByIterations.put(600,0.028);
      // Iteration 800 on choisie un learning rate 0.006
        learningRateByIterations.put(800,0.006);
      // Iteration 1000 on choisie un learning rate 0.001
        learningRateByIterations.put(1000,0.001);
        // Initialisation de l'erreur quadratique
        double quadraticError=0.0005;
        // Initialisation de la valeur du momentum
        double momentum=0.9;
       // Instanciation d'un Ramdom avec comme valeur de depart specifié la valeur de seed
        Random randomGenNum=new Random(seed);



      System.out.println("---------------------Chargement et conversion du dataset ----------------------------------------------------------");
        // Affichage des infos sur le chargement de données et de leur vectorisation
        logger.info("Data Load and vectorisation");
      // chemin de stockages des données d'entrainement
        File trainDataFile=new File(basePath+"/training");
      // Division du jeu d'entrainement
        FileSplit trainFileSplit=new FileSplit(trainDataFile, NativeImageLoader.ALLOWED_FORMATS,randomGenNum);
        ParentPathLabelGenerator labelMarker=new ParentPathLabelGenerator();
        // Conversion du jeu d'entrainement
        ImageRecordReader trainImageRecordReader=new ImageRecordReader(height,width,channels,labelMarker);
        // initialisation apres conversion
        trainImageRecordReader.initialize(trainFileSplit);
        int labelIndex=1;
      // Vectorisation du jeu d'entrainement
        DataSetIterator trainDataSetIterator=new RecordReaderDataSetIterator(trainImageRecordReader,batchSize,labelIndex,outputNum);
       // Normalisation avec une loi normale centre en 0 er reduite 1
        DataNormalization scaler=new ImagePreProcessingScaler(0,1);
        scaler.fit(trainDataSetIterator);
        trainDataSetIterator.setPreProcessor(scaler);

        // Meme chose pour le test sans normalisation
        File testDataFile=new File(basePath+"/testing");
        FileSplit testFileSplit=new FileSplit(testDataFile, NativeImageLoader.ALLOWED_FORMATS,randomGenNum);
        ImageRecordReader testImageRecordReader=new ImageRecordReader(height,width,channels,labelMarker);
        testImageRecordReader.initialize(testFileSplit);
        DataSetIterator testDataSetIterator=new RecordReaderDataSetIterator(testImageRecordReader,batchSize,labelIndex,outputNum);
        trainDataSetIterator.setPreProcessor(scaler);



      System.out.println("---------------------Configuation du model  -----------------------------------------------------------------------");
        // Configuration du jeu de neuronnes a convolution
        logger.info("Neural Network Model Configuation");
        // Configuration du reseau multi-couche
        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(seed)
          // Passage de la valeur de l'erreur quadratique
                .l2(quadraticError)
          // Optimisation avec la descente de gradient stochastique
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
          // Passage de chaque valeur du taux d'apprentissage sur les iterations definies
                .updater(new Nesterovs(new MapSchedule(ScheduleType.ITERATION,learningRateByIterations),momentum))
          // Initialisation des poids avec XAVIER
                .weightInit(WeightInit.XAVIER)
          // liste des couches
                .list()
          // premiere couche de convolution : noyau 3, canneau 1, pas de decalage 1 et la sortie est 20, activation RELU
                    .layer(0,new ConvolutionLayer.Builder()
                            .kernelSize(3,3)
                            .nIn(channels)
                            .stride(1,1)
                            .nOut(20)
                            .activation(Activation.RELU).build())
          // premiere couche de pooling
                     .layer(1, new SubsamplingLayer.Builder()
                       // type de pooling MaxPooling
                             .poolingType(SubsamplingLayer.PoolingType.MAX)
                       // noyau 2
                             .kernelSize(2,2)
                       // pas de decalage 2
                             .stride(2,2)
                             .build())
          // deuxieme couche de convolution : noyau 3, pas de decalage 2, sortie 50, activation relu
                    .layer(2, new ConvolutionLayer.Builder(3,3)
                            .stride(1,1)
                            .nOut(50)
                            .activation(Activation.RELU)
                            .build())
          // deuxieme couche de pooling
                    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                      // noyau 2
                            .kernelSize(2,2)
                      // Pas de decalage 2
                            .stride(2,2)
                            .build())
          // la couche dense : activation relu, sortie 500
                    .layer(4, new DenseLayer.Builder()
                            .activation(Activation.RELU)
                            .nOut(500)
                            .build())
          // la couche de sortie : avec comme fonction de perte qui est ici une estimation du maximum de la probabilite de la bonne sortie
                    .layer(5,new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                      // activation softmax
                            .activation(Activation.SOFTMAX)
                      // nombre de sortie
                            .nOut(outputNum)
                            .build())
          // parametrage de la dimension d'entree
                    .setInputType(InputType.convolutionalFlat(height,width,channels))
          // utilisation de la retropropagation standard
                .backpropType(BackpropType.Standard)
                .build();
        // Affichage de la configuration en JSON
        System.out.println(configuration.toJson());




      System.out.println("---------------------Creation du modéle et initialisation du modéle ----------------------------------------------------------------------");
        // Creation du reseau de convolution
        MultiLayerNetwork model=new MultiLayerNetwork(configuration);
        // initialisation du modéle
        model.init();

      System.out.println("---------------------Creation du service d'administration graphique en ligne ----------------------------------------------------------");
      // Creation de l'instance  du service d'administration graphique en ligne de notre model
        UIServer uiServer=UIServer.getInstance();
      // Gestion du stockage de notre modéle
        StatsStorage statsStorage=new InMemoryStatsStorage();
      // stockage du service en lui passant en parametre l'objet stockage
        uiServer.attach(statsStorage);
        model.setListeners(new StatsListener(statsStorage));
      // affichage du nombre de parametres du modéle
        logger.info("Total params:"+model.numParams());


      System.out.println("---------------------Entrainement du modele en parcourant les iterations ----------------------------------------------------------");
        // Entrainement du modéle
        for (int i = 0; i < epochCount; i++) {
            model.fit(trainDataSetIterator);
            logger.info("End of epoch "+i);

          System.out.println("---------------------Evaluation et Prediction  du modéle ----------------------------------------------------------");
            // Evaluation de modele
            Evaluation evaluation=model.evaluate(testDataSetIterator);
            logger.info(evaluation.stats());
            trainDataSetIterator.reset();
            testDataSetIterator.reset();
        }


      System.out.println("---------------------Sauvegarde du  modéle ----------------------------------------------------------");
        // Sauvegarde du modéle
        logger.info("Saving model ....");
        ModelSerializer.writeModel(model,new File(basePath+"/model.zip"),true);

    }
}
