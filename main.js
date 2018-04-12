const BATCH_SIZE = 2
const EPOCHS     = 400

let mobnet = {}
let voteModel = {}

init()

async function init() {
  document.querySelector('#train').addEventListener('click', async () => {
    document.querySelector('.notification').style.display = "block";
    train()
  })
  document.querySelector('#predict').addEventListener('click', async () => {
    predict()
  })
}

async function train() {
  const trainingData = getData()
  mobnet = await getMobnet()
  voteModel = getModel()

  setStatus("Training...")

  await voteModel.fit(trainingData.x, trainingData.y, {
    batchSize: BATCH_SIZE,
    epochs: EPOCHS,
    callbacks: {
      onEpochBegin: async count => console.log(`${count}/${EPOCHS}`),
      onTrainEnd:   async () => setStatus("Training Complete!")
    }
  })
}

async function predict() {
  displayImagePreview()

  const mobnetInput = imageToInput(image)
  const mobnetOutput = mobnet.predict(mobnetInput)
  const voteOutput = voteModel.predict(mobnetOutput)
  
  console.log(voteOutput)
  alert(await getResult(voteOutput))
}

function displayImagePreview() {
  const urlInput = document.querySelector('#fileUrl')
  const image = document.querySelector('#preview')
  image.src = `./public/ballots/${urlInput.value}.png`
}

async function getResult(result) {
  const THRESHOLD = 0.9
  const CHOICES = ["Duterte", "Miriam", "Binay"]

  return (await result.data()).map(v => v > THRESHOLD).reduce((resultString, isVoted, index) => {
    return isVoted ? `${resultString} ${CHOICES[index]}` : resultString
  }, "Voted for:")
}

function getModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      tf.layers.dropout({rate: 0.1}),
      tf.layers.dense({
        units: 256,
        activation: 'tanh',
      }),
      tf.layers.dropout({rate: 0.2}),
      tf.layers.dense({
        units: 256,
        activation: 'tanh',
      }),
      tf.layers.dropout({rate: 0.2}),
      tf.layers.dense({
        units: 3,
        activation: 'sigmoid'
      })
    ]
  })

  model.compile({loss: 'categoricalCrossentropy', optimizer: 'adam'})

  return model
}

async function getMobnet() {
  const MOBILENET_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
  const mobileNet     = await tf.loadModel(MOBILENET_URL)
  const mobnetOutput  = mobileNet.getLayer('conv_pw_13_relu')
  const mobnetModel   = tf.model({inputs: mobileNet.inputs, outputs: mobnetOutput.output})

  return mobnetModel
}

function getData() {
  return tf.tidy(() => {
    const trainingData = {x: [], y: []}
    const rawData = getRawData()
  
    trainingData.y = tf.tensor(rawData.y)
    trainingData.x = rawData.x.map(rawXToImage)
    trainingData.x = trainingData.x.map(imageToInput)
    trainingData.x = trainingData.x.map(inputToPredictedInput)
    trainingData.x = trainingData.x.reduce((p, c) => p.concat(c))
  
    return trainingData
  })
}

function rawXToImage(rawX) {
    const image = new Image(500, 500);
    image.src = `./public/ballots/${rawX}.png`; 

    return image
}

function imageToInput(image) {
  return tf.tidy(() => {
    const tfImage = tf.fromPixels(image)
    const resizedImage = tf.image.resizeBilinear(tfImage, [224, 224])
    const batchedImage = resizedImage.expandDims(0)
    const normalizedImage = batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1))

    return normalizedImage
  })
}

function inputToPredictedInput(input) {
  return mobnet.predict(input)
}

function getRawData() {
  const OBSERVATIONS = 10
  const XLIST = ['000', '001', '010', '011', '100', '101', '110', '111']
  const YLIST = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]

  return {
    x: XLIST.reduce((xs, x) => [...xs, ...generateBatches(x, OBSERVATIONS)], []),
    y: YLIST.reduce((ys, y) => [...ys, ...generateDuplicates(y, OBSERVATIONS)], [])
  }
}

function generateBatches(pattern, count, data=[]) {
  return count === 0 ? data : generateBatches(pattern, count-1, [...data, `Ballot-${pattern}-${count}.png`])
}

function generateDuplicates(target, count, data=[]) {
  return count === 0 ? data : generateDuplicates(target, count-1, [...data, target])
}

async function setStatus(status) {
  document.querySelector('#status').innerHTML = `${status}`
}