import os
import sys
import string
import argparse


class ParserForIEMOCAP:
    """!
    @brief Parser for the IEMOCAP dataset, finding all the audio
           utterances and corresponding annotations, based on the
           dataset structure.
    """

    def __init__(self, data_path):
        self.data_path = data_path
        self.wav_files_path = self.data_path + "Session%d/sentences/wav/"
        self.annotation_path = self.data_path + "Session%d/dialog/" \
                                                "EmoEvaluation/"
        self.dialogs_path = self.data_path + "Session%d/dialog/wav/"

        # list sessions existing in data_path
        self.sessions = [int(folder.replace('Session', '')) for folder in os.listdir(self.data_path)
                         if folder.startswith('Session')]

        self.emotion_mapping = {"neu": "neutral", "sad": "sad", "ang":
            "angry", "hap": "happy", "exc": "happy",
                                "fru": "frustrated"}
        self.gender_mapping = {"F": "female", "M": "male"}

    def get_utterances_per_dialog(self, session):
        """!
        @brief Get all the utterances for each dialog inside a
               specific session.

        @param session (\a int) Integer from 1 to 5, defining the
               session we are interested in.

        @returns \b dialog_to_utterances (\a dict) Dictionary which
                 contains all the dialogs in the session and maps each
                 dialog to the corresponding utterances.
        """
        dialog_to_utterances = {}
        for root, dirs, files in os.walk(self.wav_files_path % session):
            if not dirs:
                dialog = os.path.basename(root)
                dialog_to_utterances[dialog] = [
                    os.path.join(root, wavfile)
                    for wavfile in files if
                    os.path.splitext(wavfile)[1] in (".wav", ".npz")]
        return dialog_to_utterances

    def get_annotation_per_utterance_per_dialog(self,
                                                dialog_to_utterances,
                                                session):
        """!
        @brief Get all the annotations for each utterance for each
               dialog inside a specific session.

        @param dialog_to_utterances (\a dict) Dictionary which contains
               all the dialogs in the session and maps each dialog to
               the corresponding utterances.
        @param session (\a int) Integer from 1 to 5, defining the
               session we are interested in.

        @returns \b dialog_to_annotated utterances (\a dict) Dictionary
                 same as dialog_to_utterances, but with extra mapping
                 each utterance to the annotation found. (E.g.
                 {'Dialog_1': {'/path/to/Utterance_1.wav': {'emotion':
                  'neu', 'valence': 2.500, ...}, ...}, ...})
        """
        dialog_to_annotated_utterances = {}
        for dialog in dialog_to_utterances:
            annotation_file = (self.annotation_path % session) + \
                              str(dialog) + ".txt"
            utterances = dialog_to_utterances[dialog]
            dialog_to_annotated_utterances[dialog] = \
                self.get_annotation_per_utterance(utterances,
                                                  annotation_file)
        return dialog_to_annotated_utterances

    def get_annotation_per_utterance(self, utterances, annotation_file):
        """!
        @brief Get the annotation for each utterance, as written in the
               annotation_file.

        @param utterances (\a list) List of utterances.
        @param annotation_file (\a str) Path of the annotation file,
               where all the input utterances can be found annotated.

        @returns \b annotation_per_utterance (\a dict) Dictionary which
                 contains all the input utterances mapped to the
                 annotation found. (E.g. {'/path/to/Utterance_1.wav':
                 {'emotion': 'neu', 'valence': 2.500, ...}, ...})
        """

        annotation_per_utterance = {}

        with open(annotation_file, "r") as annfile:
            annfile_list = annfile.read().split()

            for utterance in utterances:
                annotation = dict(emotion=None, valence=None,
                                  activation=None, dominance=None)
                utterance_idx = annfile_list.index(
                    os.path.splitext(os.path.basename(utterance))[0])
                emotion_label_idx = utterance_idx + 1
                val_label_idx = utterance_idx + 2
                act_label_idx = utterance_idx + 3
                dom_label_idx = utterance_idx + 4
                start_time_idx = utterance_idx - 3
                end_time_idx = utterance_idx - 1

                annotation["emotion"] = annfile_list[emotion_label_idx]
                annotation["valence"] = annfile_list[val_label_idx] \
                    .strip(string.punctuation)
                annotation["activation"] = annfile_list[act_label_idx] \
                    .strip(string.punctuation)
                annotation["dominance"] = annfile_list[dom_label_idx] \
                    .strip(string.punctuation)
                annotation["start"] = annfile_list[start_time_idx] \
                    .strip(string.punctuation)
                annotation["end"] = annfile_list[end_time_idx] \
                    .strip(string.punctuation)
                annotation_per_utterance[utterance] = annotation

        return annotation_per_utterance

    def get_annotations(self):
        """!
        @brief Get the annotation for each utterance for each dialog
               inside each session.

        @returns \b annotations (\a dict) Dictionary in the following
                 format: {'Session1': {'Dialog_1':
                 {'/path/to/Utterance_1.wav': {'emotion': 'neu',
                  'valence': 2.500, ...}, ...}, ...}, ...})
        """

        annotations = {}

        for session in self.sessions:
            dialog_to_utterances = \
                self.get_utterances_per_dialog(session)
            annotations["Session%d" % session] = \
                self.get_annotation_per_utterance_per_dialog(
                    dialog_to_utterances, session)

        return annotations

    def convert_annotations_in_audio_hierarchy(self, annotations):
        annotated_utterances = {}

        for session in annotations:
            for dialog in annotations[session]:
                for utterance in annotations[session][dialog]:
                    annotation = annotations[session][dialog][utterance]
                    emotion = self.map_emotion(annotation["emotion"])
                    speaker_id, gender, _ = self.find_speaker_details(
                        utterance)
                    annotated_utterances[utterance] = {
                        "emotion": emotion,
                        "fold": session,
                        "speaker_id": speaker_id}

        return annotated_utterances

    def map_emotion(self, emotion_label):
        """!
        @brief Map emotion label using the self.emotion_mapping
               dictionary.
        """
        if emotion_label in self.emotion_mapping:
            emotion_label = self.emotion_mapping[emotion_label]
        else:
            emotion_label = "other"
        return emotion_label

    def map_continuous_label(self, label, mapping_dict):
        """!
        @brief Map continuous label using the corresponding
               mapping dictionary.
        """
        label_rounded = int(10 * round(float(label), 1))
        try:
            label_mapped = [mapping_dict[value_range]
                            for value_range in mapping_dict
                            if label_rounded in value_range][0]
            return label_mapped
        except:
            return

    def find_speaker_details(self, utterance_path):
        """!
        @brief Find speaker details, i.e speaker ID and gender.
        """
        utterance_name = os.path.splitext(os.path.basename(
            utterance_path))[0]
        gender = utterance_name[-4]
        speaker_id = utterance_name[0:5] + gender
        channel = ("speaker1" if (utterance_name[5] == gender) else "speaker2")
        gender = self.gender_mapping[gender]
        return speaker_id, gender, channel

    def run_parser(self):
        annotations = self.get_annotations()
        annotated_utterances = \
            self.convert_annotations_in_audio_hierarchy(annotations)

        return annotated_utterances


def parse_arguments():
    """!
    @brief Parse Arguments for parsing the IEMOCAP dataset.
    """
    args_parser = argparse.ArgumentParser(description="Parser for the "
                                                      "IEMOCAP dataset")
    args_parser.add_argument('-i', '--input', required=True,
                             help="Path of the IEMOCAP dataset")
    return args_parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()
    data_path = args.input
    parser = ParserForIEMOCAP(data_path)
    annotated_utterances = parser.run_parser()
    import pdb; pdb.set_trace()
