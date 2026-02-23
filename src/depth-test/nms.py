# https://gist.github.com/otmb/a6867efe8a93b82ef650ce3d20f94b21
import coremltools as ct

classLabels = ('sphere',)
iouThreshold = 0.6

def createNmsModelSpec(modelSpec):
    '''
    Create a coreml model with nms to filter the results of the model
    '''
    nmsSpec = ct.proto.Model_pb2.Model()
    nmsSpec.specificationVersion = ct.target.iOS18

    # Define input and outputs of the model
    for i in range(2):
        modelOutput = modelSpec.description.output[i].SerializeToString()
        nmsSpec.description.input.add()
        nmsSpec.description.input[i].ParseFromString(modelOutput)

        nmsSpec.description.output.add()
        nmsSpec.description.output[i].ParseFromString(modelOutput)

    nmsSpec.description.output[0].name = "confidence"
    nmsSpec.description.output[1].name = "coordinates"

    nms = nmsSpec.nonMaximumSuppression
    nms.confidenceInputFeatureName = "confidence"
    nms.coordinatesInputFeatureName = "coordinates"
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThreshold = iouThreshold
    nms.stringClassLabels.vector.extend(classLabels)

    return nmsSpec
