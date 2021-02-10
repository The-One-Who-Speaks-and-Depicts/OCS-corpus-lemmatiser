# OCS-corpus-lemmatiser
Lemmatiser for the Old Church Slavonic corpus. The base model is Keras implementation of seq2seq with attention, it is enhanced with generated during the training phase dictionary of token/lemma pairs, and

# How to install

* Clone the repository with terminal:

		git clone  https://github.com/The-One-Who-Speaks-and-Depicts/OCS-corpus-lemmatiser.git

* Required packages may be installed with:

		pip3 install -r requirements.txt

# How to use

## How to train

During the training phase, model generates **folder**/**name**.h5 file with its weights, **folder**/lemmatized\_**name**.txt file with dictionary of token/lemma pairs, and **folder**/train\_**name**.txt file with internal n-gram dictionary. These files should remain in the folder for prediction and accuracy scoring phases.

	python main.py −−data <path> −−join <0 or 1> −−name <string> −−grams <0 or positive_integer> −−lemma_split <0 or 1> −−stemming <0 or 1> −−folder <path> −−epochs <positive_integer> −−batch <positive_integer> −−dim <positive_integer> −−optimizer <string> −−loss <string> −−activation <string> −−early_stopping <-1 or positive_integer>

* **data**: path to the file that contains golden standard data for training phase. Data should be in [Universal Depedencies CoNLL-U format](https://universaldependencies.org/format.html). Required parameter.
* **join**: determines, whether lemmas are determined for token, or for token and its previously determined part of speech. By default, set to 0 (tokens and their respective parts of speech are not joined into one string before training). 
* **name**: specifies the name the one wants to give to the model. Used in files, saved as a result of training phase. Default model is called 'seq2seq'.
* **grams**: determines n, by which tokens are split into n-grams. Default value is 0, meaning that tokens are not split into n-grams at all.
* **lemma_split**: specifies, whether lemmas are split by n-grams, when tokens are. Default value is 1, meaning that model trains to transform n-gram of token into n-gram of lemma.
* **stemming**: determines, whether the model is trained to transform the whole token into the whole lemma, or only their parts which differ. By default, this naive stemming is not conducted, value is 0.
* **folder**: sets path, where the generated files are going to be saved. By default, it is a directory, where main.py is located.
* **epochs**: specifies the number of epochs for which the model is going to be trained. By default, set to 40.
* **batch**: determines the batch size during the training. Default batch size is 256.
* **dim**: specifies the number of latent dimensions in model layers. By default is set to 256.
* **optimizer**: optimizer, used in the model. The one that is used  by default is RMSProp, the other options are user-made optimizers, if provided with the corresponding name and python code, and keras [built-ins](https://keras.io/api/optimizers/).
* **loss**: specifies the loss function of the model. The default one is categorical crossentropy, the other possibilities, provided by Keras, are presented [here](https://keras.io/api/losses/). A user may implement their own loss functions.
* **activation**: determines the activation function of the model, softmax by default. One's own functions, as well as [other opportunities from Keras](https://keras.io/api/layers/activations/), are an option as well.
* **early_stopping**: sets the number of epochs, after which model stops training, if the loss function result values are stabilising. Default value is -1, which means that there is not going to be early stopping at all.

## How to score

The results of precision measurements of the model are shown in the terminal. The outliers for each metric used are typed into the **folder**/errors\_**name**.csv file.

	python main.py −−modus accuracy −−data <path> −−join <0 or 1> −−name <string> −−grams <0 or positive_integer> −−lemma_split <0 or 1> −−stemming <0 or 1> −−folder <path> −−dim <positive_integer> −−optimizer <string> −−loss <string> −−activation <string> −−forming_priority <back or forward>

* **data**: path to the file that contains test data. Data should be in [Universal Depedencies CoNLL-U format](https://universaldependencies.org/format.html). Required parameter.
* **name**: name of the previously trained model, that one wants to score accuracy of. The default name is 'seq2seq'.
* **folder**: sets path to the folder, where the file with outliers is going to be output, and where the model files are located.
* **forming_priority**: if the tokens are split by n-grams, the parameter determines, whether the final prediction is going to be formed as (n)y + (e)t + (on) = neon, or (ne) + (o)r + (n)a = neon. 
* **join**, **grams**, **lemma_split**, **stemming**, **dim**, **oprimizer**, **loss** and **activation** should match training parameters of model **name**.

## How to predict

The results of prediction are recorded into the file, from where the tokens are taken.

	python main.py −−modus prediction −−data <path> −−join <0 or 1> −−name <string> −−grams <0 or positive_integer> −−lemma_split <0 or 1> −−stemming <0 or 1> −−folder <path> −−dim <positive_integer> −−optimizer <string> −−loss <string> −−activation <string> −−forming_priority <back or forward>

* **data**: path to the file that contains object of a specific type [see pp.10-11](https://iling-ran.ru/web/sites/default/files/conferences/2020/2020_lingforum_abstracts.pdf) in JSON format. Required parameter.
* **name**: name of the previously trained model, that one wants to score accuracy of. The default name is 'seq2seq'.
* **folder**: sets path to the folder, where the file with outliers is going to be output, and where the model files are located.
* **forming_priority**: if the tokens are split by n-grams, the parameter determines, whether the final prediction is going to be formed as (n)y + (e)t + (on) = neon, or (ne) + (o)r + (n)a = neon. 
* **join**, **grams**, **lemma_split**, **stemming**, **dim**, **oprimizer**, **loss** and **activation** should match training parameters of model **name**.
