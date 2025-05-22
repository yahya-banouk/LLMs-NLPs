import os
from llama_cpp import Llama

model_path = os.path.expanduser("~/models/nous-hermes/Nous-Hermes-2-Mistral-7B-DPO.Q4_K_M.gguf")

llm = Llama(
    model_path=model_path,
    n_ctx=4096,
    n_threads=4  # Adjust based on your CPU cores
)

# Define your zero-shot prompt
resume_text = """
I n g é n i e u r F u l l s t a c k
A W S J a v a / J a v a S c r i p t
En tant que ingénieur logiciel fullstack , j'ai accumulé une expertise technique solide à
travers des stages et des missions, axés sur le développement web et cloud AWS. Ces
expériences m'ont permis de mettre en pratique mes compétences tout en assimilant
des connaissances concrètes du domaine. À la recherche de nouvelles perspectives,
mon objectif est de construire une carrière épanouissante dans le domaine du
développement logiciel. Mon engagement envers l'excellence et ma passion pour
Banouk Yahya
l'innovation sont mes moteurs pour apporter une contribution significative aux futurs
projets, et pour aborder avec enthousiasme de nouveaux défis qui se présentent.
C o n t a c t E x p e r i e n c e s
N° Téléphone: 2023 - 2024Expersi l Paris, France
+33758685229 Ingénieur Logiciel Fullstack
Développement, d’un outils, autour de l’IA en mode “stealth” avec AWS (bedrock,
Email: cloudfront, api getway, lambda, route53, aws S3, SES, cognito), Java (Spring boot)
Javascript (reactJs), LLM,
yahya.banouk@hotmail.com
Adoption de framework agile SCRUM.
Address Gestion du code source avec Git et GitLab.
Paris, France
2022 - 2023Zsoft Consulting l Paris La defense, France
LinkedIn: Ingénieur Logiciel Fullstack
Développement, maintenance et intégration de pas mal de modules développait from
https://www.linkedin.com/in/yahya-banouk/
scratch d'un projet interne SIRH (ReactJs, Spring Boot, Liquibase, AWS, et plain d'autres
techs.)
Adoption de framework SCRUM.
E d u c a t i o n
Gestion du code source avec Git et GitLab.
2021 - 2022 HibaPower l Kenitra, Maroc, stage
Ingénieur IoT : logiciels et Analytics Ingénieur Logiciel Fullstack et Product Owner
ENSIAS: Ecole National Supérieur Développement 'from scratch' d'un projet interne de gestion des candidatures (Web et
d'Informatique et Analyse des Systèmes Desktop avec .net c#, wpf. react js. spring entre autres) avec la mise en place de
l'infrastructure physique
Licence Mathematique informatique Adoption de la méthodologie SCRUM
Université Ibn Tofail Maroc Gestion du code source avec Git et GitHub
2020 - 2021 ORBILAC l Souk Larbaa, Maroc , stage
Administrateur Linux et Développeur Fullstack
C o m p é t e n c e s
Développement 'from scratch' d'un système interne de gestion de stock avec deux
version Web et Desktop (.net c#, wpf, react js. spring ...)
Java 8 Docker Gestion de droit d'accès, ressources, Linux
Gestion du code source avec Git et GitHub
Spring boot AWS
P r o j e t s
JavaScript / TS ReactJs
Elderly Care
.NET c# Pair-prog
Système IoT dédié à la surveillance des
Projet "Gloody" - Système
Ant design Liquibase personnes âgées vivant en solitaire, en
d'information des ressources
vue de détecter et signaler les situations
TDD humaines (SIRH), déployé et
Logs d'urgence telles que les chutes ou les
développé en utilisant un large
incendies aux parties concernées,
Swagger JMeter
éventail de technologies telles que
garantissant ainsi leur bien-être et leur
Spring Boot, ReactJS, Liquibase et
GIT/GitLab SQL sécurité,
AWS, ELK Stack, entre autres. (reactJs, react native, spring, firebase ...)
ELK Stack SCRUM
C e r t i f i c a t i o n s
A W S C l o u d P r a c t i t i o n e r
L a n g u e s
A W S A r c h i t e c t A s s o c i a t e
Anglais
Plus de 30 autres miniprojets sont disponibles sur mon profil GitHub
Francais https://github.com/yahya-banouk
I n g é n i e u r F u l l s t a c k
A W S J a v a / J a v a S c r i p t
En tant que ingénieur logiciel fullstack , j'ai accumulé une expertise technique solide à
travers des stages et des missions, axés sur le développement web et cloud AWS. Ces
expériences m'ont permis de mettre en pratique mes compétences tout en assimilant
des connaissances concrètes du domaine. À la recherche de nouvelles perspectives,
mon objectif est de construire une carrière épanouissante dans le domaine du
développement logiciel. Mon engagement envers l'excellence et ma passion pour
Banouk Yahya
l'innovation sont mes moteurs pour apporter une contribution significative aux futurs
projets, et pour aborder avec enthousiasme de nouveaux défis qui se présentent.
C o n t a c t E x p e r i e n c e s
N° Téléphone: 2023 - 2024Expersi l Paris, France
+33758685229 Ingénieur Logiciel Fullstack
Développement, d’un outils, autour de l’IA en mode “stealth” avec AWS (bedrock,
Email: cloudfront, api getway, lambda, route53, aws S3, SES, cognito), Java (Spring boot)
Javascript (reactJs), LLM,
yahya.banouk@hotmail.com
Adoption de framework agile SCRUM.
Address Gestion du code source avec Git et GitLab.
Paris, France
2022 - 2023Zsoft Consulting l Paris La defense, France
LinkedIn: Ingénieur Logiciel Fullstack
Développement, maintenance et intégration de pas mal de modules développait from
https://www.linkedin.com/in/yahya-banouk/
scratch d'un projet interne SIRH (ReactJs, Spring Boot, Liquibase, AWS, et plain d'autres
techs.)
Adoption de framework SCRUM.
E d u c a t i o n
Gestion du code source avec Git et GitLab.
2021 - 2022 HibaPower l Kenitra, Maroc, stage
Ingénieur IoT : logiciels et Analytics Ingénieur Logiciel Fullstack et Product Owner
ENSIAS: Ecole National Supérieur Développement 'from scratch' d'un projet interne de gestion des candidatures (Web et
d'Informatique et Analyse des Systèmes Desktop avec .net c#, wpf. react js. spring entre autres) avec la mise en place de
l'infrastructure physique
Licence Mathematique informatique Adoption de la méthodologie SCRUM
Université Ibn Tofail Maroc Gestion du code source avec Git et GitHub
2020 - 2021 ORBILAC l Souk Larbaa, Maroc , stage
Administrateur Linux et Développeur Fullstack
C o m p é t e n c e s
Développement 'from scratch' d'un système interne de gestion de stock avec deux
version Web et Desktop (.net c#, wpf, react js. spring ...)
Java 8 Docker Gestion de droit d'accès, ressources, Linux
Gestion du code source avec Git et GitHub
Spring boot AWS
P r o j e t s
JavaScript / TS ReactJs
Elderly Care
.NET c# Pair-prog
Système IoT dédié à la surveillance des
Projet "Gloody" - Système
Ant design Liquibase personnes âgées vivant en solitaire, en
d'information des ressources
vue de détecter et signaler les situations
TDD humaines (SIRH), déployé et
Logs d'urgence telles que les chutes ou les
développé en utilisant un large
incendies aux parties concernées,
Swagger JMeter
éventail de technologies telles que
garantissant ainsi leur bien-être et leur
Spring Boot, ReactJS, Liquibase et
GIT/GitLab SQL sécurité,
AWS, ELK Stack, entre autres. (reactJs, react native, spring, firebase ...)
ELK Stack SCRUM
C e r t i f i c a t i o n s
A W S C l o u d P r a c t i t i o n e r
L a n g u e s
A W S A r c h i t e c t A s s o c i a t e
Anglais
Plus de 30 autres miniprojets sont disponibles sur mon profil GitHub
"""

prompt = f"""
Extract the following structured data from the resume below:

Fields:
- name
- email
- phone
- education (list of degrees and institutions)
- experience (list of roles, companies, dates, descriptions)
- projects (list of project names and descriptions)
- certifications (list of certifications)
- languages (list of languages)
- location
- skills (list of skills)

Respond in JSON format.

Resume:
\"\"\"{resume_text}\"\"\"
"""

# Run the model
output = llm(prompt, max_tokens=512)

# Print result
print(output["choices"][0]["text"].strip())
