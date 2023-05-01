import { useCallback, useEffect, useState } from 'react'
import styles from '../styles/home.module.css'
import styles_button from '../components/Button.module.css'
import Link from 'next/link'
import React from 'react'

function throwError() {
  console.log(
    // The function body() is not defined
    document.body()
  )
}

const MyButton = React.forwardRef(({ onClick, href }, ref) => {
  return (
    <button href={href} onClick={onClick} ref={ref} className={styles_button.btn}>
      Essayer l'assistant
    </button>
  )
})


function Home() {
  const [count, setCount] = useState(0)
  const increment = useCallback(() => {
    setCount((v) => v + 1)
  }, [setCount])

  useEffect(() => {
    const r = setInterval(() => {
      increment()
    }, 1000)

    return () => {
      clearInterval(r)
    }
  }, [increment])

  return (
    <main className={styles.main}>
      <h1>LegalAI</h1>
      <p>
      Bienvenue sur LegalAI, une plateforme open source en cours de développement qui vise à fournir des conseils juridiques accessibles et fiables à tous.
      Notre objectif est de créer une plateforme de conseil juridique basée sur l'intelligence artificielle qui utilise des modèles de langage pour répondre à une variété de questions juridiques courantes.
      </p>
      <hr className={styles.hr} />
      <h2>Comment ça marche ?</h2>
      <p>
      LegalAI utilise des modèles de langage pour répondre à des questions juridiques. <br />
      Les modèles de langage sont des algorithmes d'apprentissage automatique qui peuvent être entraînés sur des données textuelles pour prédire des mots ou des phrases.<br />
      LegalAI utilise des modèles de langage pré-entraînés pour répondre à des questions juridiques.<br />
      LegalAI rafine son modèle de langage en utilisant des données juridiques spécifiques à la France. LegalAI est actuellement en cours de développement et ne peut pas encore répondre à toutes les questions juridiques.<br />
      </p>
      <hr className={styles.hr} />

      <Link href="/chat" passHref legacyBehavior>
      <MyButton />
      </Link>

    </main>
  )
}

export default Home
