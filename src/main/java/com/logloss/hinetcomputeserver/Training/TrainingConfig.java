package com.logloss.hinetcomputeserver.Training;

public class TrainingConfig {

    private String datasetName;
    private int epochs;
    private int batchSize;

    public TrainingConfig(String datasetName, int epochs, int batchSize) {
        this.datasetName = datasetName;
        this.epochs = epochs;
        this.batchSize = batchSize;
    }

    public String getDatasetName() {
        return datasetName;
    }

    public void setDatasetName(String datasetName) {
        this.datasetName = datasetName;
    }

    public int getEpochs() {
        return epochs;
    }

    public void setEpochs(int epochs) {
        this.epochs = epochs;
    }

    public int getBatchSize() {
        return batchSize;
    }

    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
}
