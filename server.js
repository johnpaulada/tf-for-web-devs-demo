const express = require('express')
const app = express()

app.use((req, res, next) => {
    res.set("Access-Control-Allow-Origin", "*")
    next()
})

// Host static files on public folder
app.use(express.static('public'))

app.listen(3000, () => console.log('Server listening on port http://localhost:3000!'))