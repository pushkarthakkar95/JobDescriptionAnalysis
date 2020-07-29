import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from collections import OrderedDict
import string

def computeTFIDF(tfBagOfWords, idfs):
    """
    Inverse document frequency (IDF) is how unique or rare a word is.
    """
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

def computeTF(wordDict, bagOfWords):
    """
    Term frequency (TF) is how often a word appears in a document, divided by how many words are there in a document.
    """
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def func(s):
    """
    Iterate through all punctuations variable and replace them with whitespace
    """
    punc_list = [".",";",":","!","?","/","\\",",","#","@","$","&",")","(","'","\"", "+"]
    new_s = ''
    for i in s:
        if i not in punc_list:
            new_s += i
        else:
            new_s += ' '
    return new_s

def cleanfileLine(text):
    """
    remove all numbers and punctuations
    """
    no_punct = func(text)
    no_num = ''.join([i for i in no_punct if not i.isdigit()])
    return " ".join(no_num.split())


def get_proper_nouns(document):
    """
    Using nltk pos tag library only get Proper nouns
    """
    proper_noun = [word for word,pos in pos_tag(word_tokenize(cleanfileLine(document))) if pos == 'NNP']
    return proper_noun

def remove_stop_words(proper_noun):
    """
    iterate over all the stop words and common job words and not append to the list if it’s a stop/common word
    """
    stopwords.words('english')
    common_job_words = ['key','job', 'experience', 'description', 'perform', 'skills', 'requirements', 'expert', 'responsibilities', 'understanding', 'desired']
    doc_without_Sw = [word for word in proper_noun if not word in stopwords.words()]
    doc_without_common = [word for word in doc_without_Sw if not word.lower() in common_job_words]
    return doc_without_common

def get_unique_words(documents):
    """
    Get a set of unique words from the text
    """
    uniqueWords = set()
    index = 1
    
    for each_document in documents:
        if index == 1:
            proper_noun = get_proper_nouns(each_document)
            document_without_sw = remove_stop_words(proper_noun)
            uniqueWords = set(document_without_sw)
        else:
            proper_noun = get_proper_nouns(each_document)
            document_without_sw = remove_stop_words(proper_noun)
            uniue_set = set(document_without_sw)
            uniqueWords = uniqueWords.union(uniue_set)
            
        index = index+1    
    return uniqueWords


def get_dict_after_stop_words(documents):
    """
    Get dictionary of all documents after removing stop/common words 
    """
    dict_all_documents = OrderedDict()
    index = 1
    for each_document in documents:
        proper_noun = get_proper_nouns(each_document)
        document_without_sw = remove_stop_words(proper_noun)
        dict_all_documents[str(index)] = document_without_sw
        index = index+1
    return dict_all_documents


def get_num_of_words_dict(uniqueWords, numberOfDocuments, dict_all_documents):
    """
    Get occurence of each word from unique words
    """
    index = 1;
    dict_num_of_words = OrderedDict()
    for each_dict in range(0, numberOfDocuments):
        numOfWords = dict.fromkeys(uniqueWords, 0)
        for word in dict_all_documents.get(str(index)):
            numOfWords[word] += 1
        dict_num_of_words["numOfWords" + str(index)] = numOfWords
        index = index+1
    return dict_num_of_words


def computer_Tf_all(dict_num_of_words, dict_all_documents):
    """
    Calulate term frequency for all documents
    """
    
    dict_tf = OrderedDict()
    for key1, key2 in zip(dict_num_of_words, dict_all_documents):
        dict_tf[key2] = computeTF(dict_num_of_words[key1], dict_all_documents[key2])
    return dict_tf

def get_idf(dict_num_of_words):
    list_of_values = []
    for k,v in dict_num_of_words.items():
        list_of_values.append(v)
    #print(list_of_values)    
    idfs = computeIDF(list_of_values)
    return idfs


def compute_all_tfidf(dict_tf, dict_num_of_words):
    idfs = get_idf(dict_num_of_words)
    merged = OrderedDict()
    for i , (k, v) in enumerate(dict_tf.items()):
        tfidf = computeTFIDF(v, idfs)
        merged[i] = tfidf
    return merged

def merge_all_tfidf(merged):
    merge_all = OrderedDict()
    for k,v in merged.items():
        merge_all.update(v)
    return merge_all   
    


documentA = """
Job Description iOS
Design and develop the worlds best in-vehicle applications for GM Infotainment systems
Responsible for the entire software development life cycle of your domain, including requirements, system design, development, deployment, and maintenance of the Infotainment software
Participate in architecture, requirements, design, code, and test case reviews
Provide clear and complete documentation based on the definition of the software development process
Collaborate with team members in software development activities using Scrum/Agile/SAFe development process
Additional Job Description
2+ years of experience programming in Kotlin, Java, or C++
2+ years of experience developing applications or middleware for mobile platforms like Android or iOS
Object-oriented software development experience with a solid grasp of algorithms and data structures
Experience with large code bases, developing entirely new code and maintaining existing code
Experience with testable software architecture JUnit, Espresso, TDD, MVVM, Clean Architecture
Experience with SCM tools like GIT, SVN or ClearCase
Experience with Agile/Scrum/SAFe development process and tools
Ability to perform diagnostic and investigate issues based on limited information
Excellent verbal and written communication skills
Creative, disciplined, strong sense of responsibility, delivery and schedule commitment
Bachelor's degree in Computer Science, Software/Computer Engineering or equivalent field
Advanced degrees preferred
Other Skills Preferred
Experience with Open Source Project development
Software development experience in Android Studio/Linux platform
Software development experience in 3D Unreal/Unity/OpenGL ES/Vulkan
Software development experience in a variety of embedded system
Experience with scripting, tool development and test automation framework
Experience with the development of automotive infotainment solutions
Experience with vehicle communication network protocols including CAN, MOST
"""

documentB = """
Responsibilities
Translate designs and wireframes into high quality code
Design, build, and maintain high performance, reusable, and reliable Java code
Ensure the best possible performance, quality, and responsiveness of the application
Identify and correct bottlenecks and fix bugs
Help maintain code quality, organization, and automatization
Required Qualifications
Strong knowledge of Android SDK, different versions of Android, and how to deal with different screen sizes
Familiarity with RESTful APIs to connect Android applications to back-end services
Strong knowledge of Android UI design principles, patterns, and best practices
Experience with offline storage, threading, and performance tuning
Ability to design applications around natural user interfaces, such as “touch”
Familiarity with the use of additional sensors, such as gyroscopes and accelerometers
Knowledge of the open-source Android ecosystem and the libraries available for common tasks
Ability to understand business requirements and translate them into technical requirements
A knack for benchmarking and optimization
Understanding of Google’s Android design principles and interface guidelines
Proficient understanding of code versioning tools, such as Git
"""

documentC = """
We’re Looking for Developers with:
You’re smart, love building things, have lots of energy, and love a startup environment
You have a Bachelor’s degree in computer science (or equivalent experience)
You have at least 3 years experience as a developer
You have a passion for great user experiences on mobile and the web and want to work with people that makes this a priority
You want to play a lead role in building a beautiful application with thousands of users
Skills You Should Have:
Experience developing Android apps
Java development
Experience with APIs over the network, i.e. HTTP, SOAP and REST
Building and releasing software
Working in a fast paced startup environment
"""

documentD ="""
Responsibilities
Write great code that conforms with best practices on iOS and Android platforms
Work collaboratively or individually within the development team to deliver custom app projects
Deliver coding expertise across all stages of the project lifecycle from concept to deployment
Keep knowledge current through independent learning and community events
Execute testing processes & implement optimizations to improve app performance
Develop with user experience in mind - making adjustments and tweaks based on research, testing, and other user touch-points
Requirements
Extensive mobile (iOS and/or Android) development experience
2 years experience working with Swift for iOS
2 years experience working with Java for Android (or Kotlin)
Experience working with CocoaPods in XCode and Gradle in Android Studio
Experience working with a variety of third party API’s & SDK’s
Experience with Git source control and JIRA
Using web services/APIs to interact between mobile apps and web-based content
Experience breaking down development tasks, estimating effort, and working within those estimates
"""

documentE ="""
As part of the Cash Mobile engineering team you’ll work with world class designers and a visionary product team to build the future of financial services. You’ll build and maintain new features used by millions of people and will collaborate closely with iOS and Server Engineers to move fast and with purpose.
You Will
Autonomously build end-to-end features for the app. Choose your technology and own your future
Refine the experience. We're always trying to achieve more output with less
Craft durable, well-tested code with an unwavering commitment to product quality
Build the tools you use. We are always on the lookout for technology that gives us superpowers
Have an meaningful impact on the future of the Cash product
Qualifications
You have:
A passion for product development and seeing results
A knack for creating readable, pragmatic code
Curiosity
An appreciation of simple, practical solutions
Eagerness to share your ideas, and openness to those of others
Technologies We Use And Teach
Kotlin
Protocol buffers
SQLite
Unit tests, snapshot tests, integration tests
Additional Information
Cash App treats all employees and job applicants equally. Every decision is based on merit, qualifications, and talent. We do not discriminate on the basis of race, religion, colour, national origin, gender, sexual orientation, age, marital status, veteran status, or disability status.
We will consider for employment qualified applicants with criminal histories in a manner consistent with each office’s corresponding local guidelines.Seniority Level
Entry level
Industry
Computer Software  Internet  Financial Services
Employment Type
Full-time
Job Functions
Engineering  Information Technology
"""

documentF ="""
Here at Rakuten Kobo Inc. we offer a casual working start-up environment and a group of friendly and talented individuals. Our employees rank us highly in terms of commitment to work/life balance. We realize that for our people to be innovative, creative and passionate they need to have healthy minds and bodies. We believe in rewarding all our employees with competitive salaries, performance based annual bonuses, stock options and training opportunities.
If you’re looking for a company that inspires passion, personal, and professional growth– join Kobo and come help us make reading lives better.
The Role
Rakuten Kobo’s Android application team is looking for a Contract Software Developer to work on
our Android Phone and Tablet reading application on Google Play. You will be joining a
development team of 4 to work on growing and enhancing our offering into a class leading
application. Kobo is constantly looking to innovate in terms of features, but we'll be counting on you to make sure those features are performant and of unmatched quality. As an Experienced Developer, you'll be expected to deliver excellent code, and cooperate closely with other devs
regarding challenges and roadblocks.
Here are some of the things we do and strongly believe in:
Every member on the team has a voice and is able to contribute to planning and designing. No ivory towers here
We follow an Agile development process and teams are encouraged to try new things in an effort to constantly improve
We work closely with Google in making sure our app meets their design standards. By addressing their concerns, we've been badged a Top Developer, and have been featured on Google Play.
We’re dedicated to crafting high quality, peer reviewed code. Refactoring isn’t a curse word to us and we include it as an integral part of our planning
QA is embedded on our development teams and are involved in projects from day one
We’re committed to ongoing learning and have regular discussions about development concepts, tools and to share new work built by other team members. Our developers get together to watch Google IO conference presentations and attend local Android conferences.
We encourage ideas and creativity from everyone in the company and have an innovation forum as well as Innovation Day, our company-wide hackathon, which brings together technical team members with business team members to build cool new features
Requirements
The Skillset:
Expert level knowledge of Java and the Android SDK
Outstanding object-oriented design, development, and coding skills are essential (5+ years of Java)
You have at least 2 years’ experience as a developer in an agile development environment
You have a knack for reviewing code and providing helpful feedback
Excellent analytical skills utilizing Computer Science fundamentals (Data Structures /Algorithms / Design Patterns)
You enjoy working on a dynamic, loosely structured and highly collaborative team
You're able to take a high level requirement and break it into smaller, more manageable pieces
You can identify flaws or weaknesses in existing code and improve it
You take business requirements into consideration when discussing development risks and opportunities
Extensive experience writing multi-threaded applications
Familiar with HTTP, web services and web technologies (JavaScript/CSS/HTML)
Bonus Points
You're able to write and maintain meaningful Unit tests
Experience writing and maintaining automated UI tests, along with dependency injection and data mocking frameworks
You have a keen eye for design
"""

documentG ="""
An extensive knowledge of at least one of the two official Android development languages: Java/Kotlin.
Aware of Vital Android SDK concepts (Fundamentals of Views/View Groups, Layouts, User input, Ways to get data from the web, Storing data, Action bars,Adapting apps for different screen sizes, Familiar with Android documentation).
Knowledge of modern Android Application Architecture Guidelines.
The advantage will be one of following knowledges:
MVVM Architecture with Android Architecture Components
Understanding of Material Design Guidelines
Git knowledge
Use Navigation Component
Clean Architecture principles
Using Coroutines to handle data flow (instead of RXJava)
SOLID
Use of Dagger
Creating network requests with Retrofit
Creating gRPC requests with protobuf
Persisting data with Room Persistence Library
UI testing with Espresso
Code quality checks using Lint and Detekt
Unit testing each layer of the app
"""

documentH ="""
We’re looking for a Senior iOS/Android developer who will take a key role on our client’s team. As a Senior iOS/Android developer, you will be an integral contributor to a team responsible for the frontend development of a world class consumer IoT product.
Key Responsibilities
iOS/Android application development using Swift/Kotlin
Design, develop and test front end software components in an Agile environment
Architect extendable and reusable solutions
Perform performance analysis and optimization
Document software designs and interfaces
Bug fixing and maintenance
Desired Skills And Experience
Expert level iOS/Android mobile development experience
2+ years’ production experience with Swift / Kotlin
2+ years’ production experience using Xcode and Android Studio
4+ years’ experience of C++, Java, C#
Experience with REST API services
Proven experience working with Agile methodology, CI/CD
Bachelor’s degree in Computer Science
Invest Ottawa and Bayview Yards partner with clients who are equal opportunity employers. We invite and welcome applications from people with disabilities. Accommodations at any point in our hiring process are available upon request.

"""

uniqueWords = get_unique_words([documentA, documentB, documentC, documentD, documentE, documentF, documentG, documentH])
without_stop_words = get_dict_after_stop_words([documentA, documentB, documentC, documentD, documentE, documentF, documentG, documentH])
numOfWordsDict = get_num_of_words_dict(uniqueWords, 8, without_stop_words)
tf_allDocuments = computer_Tf_all(numOfWordsDict, without_stop_words)
merged = merge_all_tfidf(tf_allDocuments)

sorted_a = OrderedDict(sorted(merged.items(), key=lambda kv: kv[1], reverse=True))
list_words = list(sorted_a.keys())
list_values = list(sorted_a.values())
top_20_keys = []
for i in range(0, 20):
    top_20_keys.append(list_words[i])


listToString = (" ").join(top_20_keys)
wc = WordCloud(width=800, height=400, max_words=200).generate(listToString)
# Display the generated image:
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.show()



