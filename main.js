const BATCH_SIZE = 20
const EPOCHS     = 100

let mobnet = {};
let voteModel = {};

init()

async function init() {
  document.querySelector('#train').addEventListener('click', async () => {
    document.querySelector('#train').classList.add('is-loading')
    train()
  })
  document.querySelector('#predict').addEventListener('click', async () => {
    document.querySelector('#preview').style.display = "block";
    predict()
  })
}

async function train() {
  mobnet = await getMobnet()

  const trainingData = await getData()
  voteModel = getModel()

  setStatus("Training...")

  await voteModel.fit(trainingData.x, trainingData.y, {
    batchSize: BATCH_SIZE,
    epochs: EPOCHS,
    callbacks: {
      onEpochBegin: async count => {
        const progress = `${Math.round(count/EPOCHS*100)}%`
        console.log(progress)
      },
      onTrainEnd:   async () => {
        document.querySelector('#train').classList.remove('is-loading')
        document.querySelector('#form').style.display = "block";
        document.querySelector('.notification').style.display = "block";

        setStatus("Training Complete!")
      }
    }
  })

  const result = voteModel.evaluate(trainingData.x, trainingData.y, {
    batchSize: BATCH_SIZE,
    epochs: EPOCHS,
  });

  console.log("Accuracy:")
  result[1].print();
}

async function predict() {
  const image = await displayImagePreview()

  const mobnetInput = imageToInput(image)
  const mobnetOutput = mobnet.predict(mobnetInput)
  const voteOutput = voteModel.predictOnBatch(mobnetOutput)

  swal(
    'You voted for:',
    await getResult(voteOutput),
    'success'
  )
}

async function displayImagePreview() {
  const urlInput = document.querySelector('#fileUrl')
  const image = document.querySelector('#preview')

  return await loadImage(image, `/public/ballots/${urlInput.value}`)
}

async function loadImage(image, src) {
  image.src = src
  image.crossOrigin = "Anonymous"
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext("2d")

  return await new Promise((resolve, reject) => {
    image.onload = () => {
      canvas.width = image.width
      canvas.height = image.height
      ctx.drawImage(image, 0, 0)
      image.src = canvas.toDataURL("image/png")
      resolve(image)
    }
  })
}

async function getResult(result) {
  const THRESHOLD = 0.9
  const CHOICES = ["Duterte", "Miriam", "Binay"]

  return (await result.data()).map(v => v > THRESHOLD).reduce((resultString, isVoted, index) => {
    return isVoted ? `${resultString} ${CHOICES[index]}` : resultString
  }, "")
}

function getModel() {
  const model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: [7, 7, 256]}),
      tf.layers.dense({
        units: 48,
        activation: 'relu',
      }),
      tf.layers.dense({
        units: 3,
        activation: 'sigmoid'
      })
    ]
  })

  model.compile({optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy']});

  return model
}

async function getMobnet() {
  const MOBILENET_URL = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json'
  const mobileNet     = await tf.loadModel(MOBILENET_URL)
  const mobnetOutput  = mobileNet.getLayer('conv_pw_13_relu')
  const mobnetModel   = tf.model({inputs: mobileNet.inputs, outputs: mobnetOutput.output})

  return mobnetModel
}

async function getData() {
  const trainingData = {x: [], y: []}
  const rawData = getRawData()
  
  trainingData.y = tf.tensor(rawData.y)
  trainingData.x = await Promise.all(rawData.x.map(rawXToImage))
  trainingData.x = trainingData.x.map(imageToInput)
  trainingData.x = trainingData.x.map(inputToPredictedInput)
  trainingData.x = trainingData.x.reduce((p, c) => p.concat(c))

  return trainingData
}

function rawXToImage(rawX) {
    const image = new Image(500, 500);
    return loadImage(image, `/public/ballots/${rawX}`)
}

function imageToInput(image) {
  return tf.tidy(() => {
    const tfImage = tf.fromPixels(image);
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