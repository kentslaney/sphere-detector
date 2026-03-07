import coremltools as ct
from .utils import dist
from .cml import config

depth_model = ct.models.MLModel(str(dist / f"{config.da2_name}.mlpackage"))
spheres_model = ct.models.MLModel(str(dist / "spheres.mlpackage"))

model = ct.utils.make_pipeline(depth_model, spheres_model)
model.save(dist / "pipelined.mlpackage")

modelSpec = model.get_spec()
classLabels = ("sphere",)
nmsSpec = ct.proto.Model_pb2.Model()
nmsSpec.specificationVersion = config.opset_version

out0, out1 = iter(modelSpec.description.output)
out0_shape = (config.candidates, 1)
out1_shape = (config.candidates, 4)
out0.type.multiArrayType.shape[:] = out0_shape
out1.type.multiArrayType.shape[:] = out1_shape

for i in range(2):
    unsuppressedOutput = modelSpec.description.output[i].SerializeToString()
    nmsSpec.description.input.add()
    nmsSpec.description.input[i].ParseFromString(unsuppressedOutput)

    nmsSpec.description.output.add()
    nmsSpec.description.output[i].ParseFromString(unsuppressedOutput)

nmsSpec.description.output[0].name = "confidence"
nmsSpec.description.output[1].name = "coordinates"

outputSizes = [len(classLabels), 4]
for i in range(len(outputSizes)):
    maType = nmsSpec.description.output[i].type.multiArrayType
    maType.shapeRange.sizeRanges.add()
    maType.shapeRange.sizeRanges[0].lowerBound = 0
    maType.shapeRange.sizeRanges[0].upperBound = -1

    maType.shapeRange.sizeRanges.add()
    maType.shapeRange.sizeRanges[1].lowerBound = outputSizes[i]
    maType.shapeRange.sizeRanges[1].upperBound = outputSizes[i]
    del maType.shape[:]

nms = nmsSpec.nonMaximumSuppression
nms.confidenceInputFeatureName = out0.name
nms.coordinatesInputFeatureName = out1.name
nms.confidenceOutputFeatureName = "confidence"
nms.coordinatesOutputFeatureName = "coordinates"
nms.iouThreshold = config.iou_threshold
nms.stringClassLabels.vector.extend(classLabels)

pipeline = ct.models.pipeline.Pipeline(
        input_features=[
            ("image", ct.models.datatypes.Array(3, *config.resolution))],
        output_features=["confidence", "coordinates"])
pipeline.spec.specificationVersion = config.opset_version

pipeline.add_model(modelSpec)
pipeline.add_model(nmsSpec)

pipeline.spec.description.input[0].ParseFromString(
    modelSpec.description.input[0].SerializeToString())
pipeline.spec.description.output[0].ParseFromString(
    nmsSpec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(
    nmsSpec.description.output[1].SerializeToString())

model = ct.models.MLModel(pipeline.spec, weights_dir=model.weights_dir)
model.save(dist / "e2e.mlpackage")
