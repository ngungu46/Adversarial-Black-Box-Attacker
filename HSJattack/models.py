import numpy as np

import tensorflow as tf
decode_predictions = tf.keras.applications.inception_v3.decode_predictions

class Model:
    def __init__(self, pretrained_model, label, dataset = 'imagenet'):
        self.count = 0
        self.pretrained_model = pretrained_model
        self.label = label

        if dataset == 'imagenet':
            self.get_labels = self.get_imagenet_labels
        elif dataset == 'butterfly':
            self.get_labels = self.get_butterflies_labels

    def get_imagenet_labels(self, probs, top = 4):
        return decode_predictions(probs, top = top)
    
    def get_butterflies_labels(self, probs, top = 4):
        batch_size = probs.shape[0]

        mapping = np.array([
            'ADONIS', 'AFRICAN GIANT SWALLOWTAIL', 'AMERICAN SNOOT', 'AN 88', 'APPOLLO', 'ARCIGERA FLOWER MOTH', 'ATALA', 'ATLAS MOTH', 'BANDED ORANGE HELICONIAN', 'BANDED PEACOCK', 'BANDED TIGER MOTH', 'BECKERS WHITE', 'BIRD CHERRY ERMINE MOTH', 'BLACK HAIRSTREAK', 'BLUE MORPHO', 'BLUE SPOTTED CROW', 'BROOKES BIRDWING', 'BROWN ARGUS', 'BROWN SIPROETA', 'CABBAGE WHITE', 'CAIRNS BIRDWING', 'CHALK HILL BLUE', 'CHECQUERED SKIPPER', 'CHESTNUT', 'CINNABAR MOTH', 'CLEARWING MOTH', 'CLEOPATRA', 'CLODIUS PARNASSIAN', 'CLOUDED SULPHUR', 'COMET MOTH', 'COMMON BANDED AWL', 'COMMON WOOD-NYMPH', 'COPPER TAIL', 'CRECENT', 'CRIMSON PATCH', 'DANAID EGGFLY', 'EASTERN COMA', 'EASTERN DAPPLE WHITE', 'EASTERN PINE ELFIN', 'ELBOWED PIERROT', 'EMPEROR GUM MOTH', 'GARDEN TIGER MOTH', 'GIANT LEOPARD MOTH', 'GLITTERING SAPPHIRE', 'GOLD BANDED', 'GREAT EGGFLY', 'GREAT JAY', 'GREEN CELLED CATTLEHEART', 'GREEN HAIRSTREAK', 'GREY HAIRSTREAK', 'HERCULES MOTH', 'HUMMING BIRD HAWK MOTH', 'INDRA SWALLOW', 'IO MOTH', 'Iphiclus sister', 'JULIA', 'LARGE MARBLE', 'LUNA MOTH', 'MADAGASCAN SUNSET MOTH', 'MALACHITE', 'MANGROVE SKIPPER', 'MESTRA', 'METALMARK', 'MILBERTS TORTOISESHELL', 'MONARCH', 'MOURNING CLOAK', 'OLEANDER HAWK MOTH', 'ORANGE OAKLEAF', 'ORANGE TIP', 'ORCHARD SWALLOW', 'PAINTED LADY', 'PAPER KITE', 'PEACOCK', 'PINE WHITE', 'PIPEVINE SWALLOW', 'POLYPHEMUS MOTH', 'POPINJAY', 'PURPLE HAIRSTREAK', 'PURPLISH COPPER', 'QUESTION MARK', 'RED ADMIRAL', 'RED CRACKER', 'RED POSTMAN', 'RED SPOTTED PURPLE', 'ROSY MAPLE MOTH', 'SCARCE SWALLOW', 'SILVER SPOT SKIPPER', 'SIXSPOT BURNET MOTH', 'SLEEPY ORANGE', 'SOOTYWING', 'SOUTHERN DOGFACE', 'STRAITED QUEEN', 'TROPICAL LEAFWING', 'TWO BARRED FLASHER', 'ULYSES', 'VICEROY', 'WHITE LINED SPHINX MOTH', 'WOOD SATYR', 'YELLOW SWALLOW TAIL', 'ZEBRA LONG WING'
        ], dtype=str)

        indices = np.argsort(probs, axis=1)
        indices = indices[:, ::-1]
        indices = indices[:, :top]

        # print(indices)

        labels_top = mapping[indices]
        probs_top = probs[np.arange(batch_size)[:, None], indices]

        out = [[(label, label, prob) for prob, label in zip(probs_row, labels_row)] \
                for probs_row, labels_row in zip(probs_top, labels_top)]
        
        # print(out)

        return out

    def get_logits(self, image):
        """
        input: normalized tensor of shape (B, H, W, C)
        output: numpy array of logits
        """

        self.count += 1

        preds = self.pretrained_model(image)
        
        return preds
    

    def predict(self, image):
        return self.get_labels(self.get_logits(image))


    def decision(self, image, verbose=False):
        # batched = (image.shape[0] != 1) and (len(image.shape) == 4)

        best = self.predict(image)
        
        if verbose:
            print(self.label)
            for i in range(3):
                print(best[i])

        out = [tup[0][0] != self.label for tup in best]

        return out