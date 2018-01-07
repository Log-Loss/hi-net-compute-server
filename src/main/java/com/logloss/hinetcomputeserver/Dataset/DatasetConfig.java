package com.logloss.hinetcomputeserver.Dataset;

import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;


public class DatasetConfig {
    private int epochs;
    private int batchSize;
    private DataSetIterator trainDataIter;
    private DataSetIterator validDataIter;
    private DataSetIterator testDataIter;

    public DatasetConfig(int epochs, int batchSize, DataSetIterator trainDataIter, DataSetIterator validDataIter, DataSetIterator testDataIter) {
        this.epochs = epochs;
        this.batchSize = batchSize;
        this.trainDataIter = trainDataIter;
        this.validDataIter = validDataIter;
        this.testDataIter = testDataIter;
    }

    public DatasetConfig(int epochs, int batchSize) {
        this(epochs, batchSize, null, null, null);
    }

    public DatasetConfig(int epochs, int batchSize, DataSetIterator trainDataIter, DataSetIterator testDataIter) {
        this(epochs, batchSize, trainDataIter, null, testDataIter);
    }

    public void setDataIter(DataSetIterator trainDataIter, DataSetIterator testDataIter) {
        this.trainDataIter = trainDataIter;
        this.testDataIter = testDataIter;
    }

    public void setDataIter(DataSetIterator trainDataIter, DataSetIterator validDataIter, DataSetIterator testDataIter) {
        this.trainDataIter = trainDataIter;
        this.validDataIter = validDataIter;
        this.testDataIter = testDataIter;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public int getEpochs() {
        return epochs;
    }

    public DataSetIterator getTestDataIter() {
        return testDataIter;
    }

    public DataSetIterator getTrainDataIter() {
        return trainDataIter;
    }

    public DataSetIterator getValidDataIter() {
        return validDataIter;
    }
}
