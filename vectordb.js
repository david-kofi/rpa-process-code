const {Pinecone} = require('@pinecone-database/pinecone')
const pc = new Pinecone({
    apiKey:"pcsk_33ucs9_DKggDFT8mT3miMN25r3VUBkaLVnCnCEqBZK2JPjizCg4DqhYtsArvbQXb4vtyjG"
})

const index = pc.index('adv-vector-research')

async function name(params) {
    
   await index.namespace().upsert(params);


}