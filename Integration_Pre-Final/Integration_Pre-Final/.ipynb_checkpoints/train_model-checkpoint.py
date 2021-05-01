{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "operating-humor",
   "metadata": {},
   "outputs": [],
   "source": [
    "# USAGE\n",
    "# python train_model.py --embeddings output/embeddings.pickle \\\n",
    "#\t--recognizer output/recognizer.pickle --le output/le.pickle\n",
    "\n",
    "# import the necessary packages\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.svm import SVC\n",
    "import argparse\n",
    "import pickle\n",
    "\n",
    "# construct the argument parser and parse the arguments\n",
    "ap = argparse.ArgumentParser()\n",
    "ap.add_argument(\"-e\", \"--embeddings\", required=True,\n",
    "\thelp=\"path to serialized db of facial embeddings\")\n",
    "ap.add_argument(\"-r\", \"--recognizer\", required=True,\n",
    "\thelp=\"path to output model trained to recognize faces\")\n",
    "ap.add_argument(\"-l\", \"--le\", required=True,\n",
    "\thelp=\"path to output label encoder\")\n",
    "args = vars(ap.parse_args())\n",
    "\n",
    "# load the face embeddings\n",
    "print(\"[INFO] loading face embeddings...\")\n",
    "data = pickle.loads(open(args[\"embeddings\"], \"rb\").read())\n",
    "\n",
    "# encode the labels\n",
    "print(\"[INFO] encoding labels...\")\n",
    "le = LabelEncoder()\n",
    "labels = le.fit_transform(data[\"names\"])\n",
    "\n",
    "# train the model used to accept the 128-d embeddings of the face and\n",
    "# then produce the actual face recognition\n",
    "print(\"[INFO] training model...\")\n",
    "recognizer = SVC(C=1.0, kernel=\"linear\", probability=True)\n",
    "recognizer.fit(data[\"embeddings\"], labels)\n",
    "\n",
    "# write the actual face recognition model to disk\n",
    "f = open(args[\"recognizer\"], \"wb\")\n",
    "f.write(pickle.dumps(recognizer))\n",
    "f.close()\n",
    "\n",
    "# write the label encoder to disk\n",
    "f = open(args[\"le\"], \"wb\")\n",
    "f.write(pickle.dumps(le))\n",
    "f.close()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
