# Hi, Net Compute Server
this is part of the compute server, needs model configuration and training configuration

### Requirements
* JDK 1.8
* Maven

### Getting Start
it needs a new configuration passed from frontend:    
**Training Configuration**
```
dataset name       eg. "mnist"
model ID           eg. 1001
number of epochs   eg. 3
batch size         eg. 32
```

the **model configuration** should be linked to the previous step

#### TODO
the **statistic information** need to be passed to another server of frontend directly.