package com.logloss.hinetcomputeserver.Dataset;

import com.logloss.hinetcomputeserver.Training.TrainingConfig;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.datasets.iterator.impl.CifarDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.IrisDataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.nio.charset.Charset;
import java.util.Random;


public class BuildInDataset {

    private DatasetConfig datasetConfig;

    public BuildInDataset(TrainingConfig trainingConfig) throws Exception {
        switch (trainingConfig.getDatasetName()) {
            case "mnist":
                datasetConfig = new DatasetConfig(trainingConfig.getEpochs(),
                        trainingConfig.getBatchSize(),
                        new MnistDataSetIterator(trainingConfig.getBatchSize(),true,12345),
                        new MnistDataSetIterator(trainingConfig.getBatchSize(),false,12345) );
                break;
            case "cifar":
                datasetConfig = new DatasetConfig(trainingConfig.getEpochs(),
                        trainingConfig.getBatchSize(),
                        new CifarDataSetIterator(trainingConfig.getBatchSize(), 50000, true),
                        new CifarDataSetIterator(trainingConfig.getBatchSize(), 10000, false));
                break;
            case "iris":
                datasetConfig = new DatasetConfig(trainingConfig.getEpochs(),
                        trainingConfig.getBatchSize(),
                        new IrisDataSetIterator(trainingConfig.getBatchSize(), 100),
                        new IrisDataSetIterator(trainingConfig.getBatchSize(), 50));
                break;
            case "eminst-complete":
                datasetConfig = new DatasetConfig(trainingConfig.getEpochs(),
                        trainingConfig.getBatchSize(),
                        new EmnistDataSetIterator(EmnistDataSetIterator.Set.COMPLETE, trainingConfig.getBatchSize(), true),
                        new EmnistDataSetIterator(EmnistDataSetIterator.Set.COMPLETE, trainingConfig.getBatchSize(), false));
                break;
            case "eminst-merge":
                datasetConfig = new DatasetConfig(trainingConfig.getEpochs(),
                        trainingConfig.getBatchSize(),
                        new EmnistDataSetIterator(EmnistDataSetIterator.Set.MERGE, trainingConfig.getBatchSize(), true),
                        new EmnistDataSetIterator(EmnistDataSetIterator.Set.MERGE, trainingConfig.getBatchSize(), false));
                break;
            case "eminst-balance":
                datasetConfig = new DatasetConfig(trainingConfig.getEpochs(),
                        trainingConfig.getBatchSize(),
                        new EmnistDataSetIterator(EmnistDataSetIterator.Set.BALANCED, trainingConfig.getBatchSize(), true),
                        new EmnistDataSetIterator(EmnistDataSetIterator.Set.BALANCED, trainingConfig.getBatchSize(), false));
                break;
            case "eminst-letter":
                datasetConfig = new DatasetConfig(trainingConfig.getEpochs(),
                        trainingConfig.getBatchSize(),
                        new EmnistDataSetIterator(EmnistDataSetIterator.Set.LETTERS, trainingConfig.getBatchSize(), true),
                        new EmnistDataSetIterator(EmnistDataSetIterator.Set.LETTERS, trainingConfig.getBatchSize(), false));
                break;
            case "eminst-digits":
                datasetConfig = new DatasetConfig(trainingConfig.getEpochs(),
                        trainingConfig.getBatchSize(),
                        new EmnistDataSetIterator(EmnistDataSetIterator.Set.DIGITS, trainingConfig.getBatchSize(), true),
                        new EmnistDataSetIterator(EmnistDataSetIterator.Set.DIGITS, trainingConfig.getBatchSize(), false));
                break;
            case "eminst-mnist":
                datasetConfig = new DatasetConfig(trainingConfig.getEpochs(),
                        trainingConfig.getBatchSize(),
                        new EmnistDataSetIterator(EmnistDataSetIterator.Set.MNIST, trainingConfig.getBatchSize(), true),
                        new EmnistDataSetIterator(EmnistDataSetIterator.Set.MNIST, trainingConfig.getBatchSize(), false));
                break;

            case "shakespeare":
                datasetConfig = new DatasetConfig(trainingConfig.getEpochs(),
                        trainingConfig.getBatchSize(),
                        getShakespeareIterator(trainingConfig.getBatchSize(), 1000),
                        getShakespeareIterator(trainingConfig.getBatchSize(), 1000));
        }
    }

    public DataSetIterator getTrainIter() {
        return this.datasetConfig.getTrainDataIter();
    }

    public DataSetIterator getTestIter() {
        return this.datasetConfig.getTestDataIter();
    }


    public static CharacterIterator getShakespeareIterator(int miniBatchSize, int sequenceLength) throws Exception{
        //The Complete Works of William Shakespeare
        //5.3MB file in UTF-8 Encoding, ~5.4 million characters
        //https://www.gutenberg.org/ebooks/100
        String url = "https://s3.amazonaws.com/dl4j-distribution/pg100.txt";
        String tempDir = System.getProperty("java.io.tmpdir");
        String fileLocation = tempDir + "/Shakespeare.txt";	//Storage location from downloaded file
        File f = new File(fileLocation);
        if( !f.exists() ){
            FileUtils.copyURLToFile(new URL(url), f);
            System.out.println("File downloaded to " + f.getAbsolutePath());
        } else {
            System.out.println("Using existing text file at " + f.getAbsolutePath());
        }

        if(!f.exists()) throw new IOException("File does not exist: " + fileLocation);	//Download problem?

        char[] validCharacters = CharacterIterator.getMinimalCharacterSet();	//Which characters are allowed? Others will be removed
        return new CharacterIterator(fileLocation, Charset.forName("UTF-8"),
                miniBatchSize, sequenceLength, validCharacters, new Random(12345));
    }
}
