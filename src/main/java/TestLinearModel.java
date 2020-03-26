import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.File;
public class TestLinearModel {
    public static void main(String[] args) throws Exception{

      System.out.println("---------------------Creation des hyperparametres du reseau ----------------------------------------------------------");
        int seed=123;
        // Taux d'apprentissage
        double learningRate=0.01;
        // Taille des echantillons
        int batchSize=50;
        // Nombre d'iterations
        int nEpochs=100;
       // Nombre d'entrees et de sorties du reseau multicouches
        int numIn=2;int numOut=1;
        // Nombre de couches cachées
        int nHidden=20;


      System.out.println("---------------------Chargement et conversion du dataset ----------------------------------------------------------");
        // Recuperation du fichier des exemples d'entrainement par son chemain comme ils se trouve dans le classpath
        String filePathTrain=new ClassPathResource("linear_data_train.csv").getFile().getPath();
        // Meme chose pour le fichier de test
        String filePathTest=new ClassPathResource("linear_data_eval.csv").getFile().getPath();
        // Comme ces fichiers CSV, on les convertit (entraienement et test) en format DataSetIterator permettant de charger
      // les données dans un reseau de neuronnes, et d'aider  à oganiser le traitement par lots
        RecordReader rr=new CSVRecordReader();
        rr.initialize(new FileSplit(new File(filePathTrain)));
        DataSetIterator dataSetTrain=new RecordReaderDataSetIterator(rr,batchSize,0,1);
        RecordReader rrTest=new CSVRecordReader();
        rrTest.initialize(new FileSplit(new File(filePathTest)));
        DataSetIterator dataSetTest=new RecordReaderDataSetIterator(rrTest,batchSize,0,1);


      System.out.println("---------------------Configuation du model  -----------------------------------------------------------------------");
        // Instanciation de la classe de configuration du reseau de neuronnes miulti-couche
        MultiLayerConfiguration configuration=new NeuralNetConfiguration.Builder()
                .seed(seed)
          // initialisation des poids du reseau avec l'initialisateur Xavier
                .weightInit(WeightInit.XAVIER)
          // cette configuration se fait avec l'optimiseur SDG en lui passant comme parametre le taux d'apprentissage
                .updater(new Sgd(learningRate))
          // Creation de la liste des couches cachées
                .list()
          // fonction de creation des couches cachées dense
                    .layer(0, new DenseLayer.Builder()
                      // nombre d'entrees aux couches de maniére sequentielle
                        .nIn(numIn)
                      //  nombre de sorties aux couches de maniere sequentielle aussi
                        .nOut(nHidden)
                      // fonction d'activation qui est RELU
                        .activation(Activation.RELU).build())
          // couche de sortie du reseau avec comme parametre la fonction de perte est ici l'entropie croisée
                    .layer(1,new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                      // nombre d'entrees aux couches de maniére sequentielle
                        .nIn(nHidden)
                      //  nombre de sorties aux couches de maniere sequentielle aussi
                        .nOut(numOut)
                      // la fonction d'activation de la couche de sortie qui est ici la fonction sigmoide
                        .activation(Activation.SIGMOID).build())
          // Construction de la configuration du reseau multicouches
                .build();


      System.out.println("---------------------Creation du modéle et initialisation du modéle ----------------------------------------------------------------------");
            // Creation du modéle de reseau multicouche en lui passant comme parametre la configuration
        MultiLayerNetwork model=new MultiLayerNetwork(configuration);
        // initaialisation du modele
        model.init();


      System.out.println("---------------------Creation du service d'administration graphique en ligne ----------------------------------------------------------");
        // Creation de l'instance  du service d'administration graphique en ligne de notre model
        UIServer uiServer=UIServer.getInstance();
        // Gestion du stockage de notre modéle
        StatsStorage statsStorage=new InMemoryStatsStorage();
        // stockage du service en lui passant en parametre l'objet stockage
        uiServer.attach(statsStorage);
        // Parametrage des listeneurs pour les requetes clientes
        model.setListeners(new StatsListener(statsStorage));

        //model.setListeners(new ScoreIterationListener(10));


      System.out.println("---------------------Entrainement du modele en parcourant les iterations ----------------------------------------------------------");
      // Parcours des iterations du model et entrainement
        for (int i = 0; i <nEpochs ; i++) {
            model.fit(dataSetTrain);
        }



      System.out.println("---------------------Evaluation et Prediction du modéle du modéle ----------------------------------------------------------");
        // Début de l'evaluation du modéle
        System.out.println("Model Evaluation....");

      // Evaluation du modéle avec comme parametre le le nombre de sortie
        Evaluation evaluation=new Evaluation(numOut);
        // Parcours de l'ensemble des donnes de test
        while(dataSetTest.hasNext()){
            DataSet dataSet=dataSetTest.next();
            // Extraction des caracteristiques du dataset dans un tableau INDArray
            INDArray features=dataSet.getFeatures();
            // Meme chose pour les cibles
            INDArray labels=dataSet.getLabels();
            // Prediction du modéle
            INDArray predicted=model.output(features,false);
            evaluation.eval(labels,predicted);
        }
// Affichage des statistiques des evaluations
        System.out.println(evaluation.stats());
        System.out.println("************************");
        System.out.println("Prédiction :");
        INDArray xs= Nd4j.create(new double[][]{
                {0.766837548998774,0.486441995062381},
                {0.332894760145352,-0.0112936854155695},
                {0.377466773756814,0.155504538357614}
        });
        // Affichage des des cibles predits
        INDArray ys=model.output(xs);
        System.out.println(ys);
    }
}
