import torch
import os
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import KMeans
import numpy as np
from model import cosine,euclidean_distance
import h5py
from compute_rouge import get_score,get_score_all
from model import createGraph,findHighestSimilarityRankTopic
# from hdbscan import HDBSCAN
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# article = ["by leon watson", "published :", "19:32 est , 24 april 2013", "|", "updated :", "17:40 est , 25 april 2013", "a 15-year old boy has been charged with the murder of a teenager stabbed to death on a bus .", "derek boateng , 16 , was attacked in broad daylight on the 393 bus in highbury new park , north london , at around 3pm on tuesday .", "he was airlifted to hospital but died yesterday .", "a police spokesman said : ` the 15-year-old boy from havering will appear in custody at highbury corner youth court accused of the murder of 16-year-old derek boateng . '", "police say boateng , who has three older sisters , was attacked following a row with another teenager .", "it was the fifth stabbing of a teenager in london in the past four days .", "the london air ambulance landed on highbury grove school 's astro turf pitch and airlifted the teenager to hospital in east london where he died yesterday .", "today , his devastated sisters paid tribute to the little brother they described as ` always laughing ' and revealed his birthday presents are still lying unopened at the family home in nearby hackney .", "derek 's oldest sister gifty , 29 , said he was ` best brother in the world .", "she told the evening standard : ' i could not have had a better brother . '", "earlier , she told how derek was doing his gcses and wanted to be an engineer .", "he said : ` he just fixes stuff , like things you ca n't figure out he comes to do .", "` he paints and plays the drums - he was very artistic .", "he had a lot of friends and knew a lot of people around the area and had friends from primary school still .", "we are completely devastated and still in shock . '", "another of his siblings , linda , 25 , said : ` all we know is that he was on a bus by himself .", "he normally gets the train .", "` from what we know , he was going to meet his friend at highbury grove when it happened .", "` he was always laughing , everything was a joke .", "you ca n't even tell him off because he just started laughing at you .", "` he was stabbed on his birthday on the 23rd and died the day after .", "he was coming home to his presents , he had n't even opened them yet . '", "she added : ` you hear about this sort of thing all the time and never think this is going to come to you .", "` knives are for the kitchen .", "put the knives away , just concentrate on your education .", "they have left the family very devastated . '", "i do n't think they understand what they are doing .", "the guy who did it is probably really scared because they do n't understand what they have done .", "` it 's ridiculous .", "killing someone and going back to your group does n't make you cool . '", "more than 2,000 people have paid tribute on facebook to a murdered teenager who died the day after being stabbed on his sixteenth birthday .", "derek boateng , who wanted to be an engineer , was fatally stabbed on a 393 bus on tuesday afternoon .", "he died in hospital last night .", "and now more than 2,700 people have flooded a tribute page set up for the tragic youngster - known as ddot - on facebook .", "the creater of the ` r.i.p derek - ddot ' page wrote : ` the thing that hurts the most is that i didnt even wish him a happy birthday ` happy birthday , r.i.p ' .", "rochelle ` nunu ' mullings said : ` extremely heartbreaking .", "i pray no more familes have to bury their kids because of violence . '", "i pray that these youngsters start to value their own lifes and that of others more .", ".", "violence solves nothing at all just creates more problems .", "put the weapons down !!", "god guide us this is terrible . '", "sarah rey rey wrote : ` my taughts are with his friends and family , another young life gone , rest in peace ' .", "sheree wright said : ` rest in perfect peace , i know the streets are tough but i 'm disgusted that young boys & girls in this world think they can play god & take someones life !", "too many mothers are burying their babies .", "` thoughts to all the family & friends for their loss .", "this page will make him proud ' .", "the teenager was said by locals to be a former student at highbury grove school .", "police were called on tuesday afternoon and the victim was rushed to hospital in a critical condition but he succumbed to his injuries yesterday evening .", "nearby residents said the boy was stabbed by a black teenager , aged around 16 , who was wearing a baseball cap and who also pushed a passenger to the ground before running off through the nearby spring gardens estate .", "andrew lloyd webber had visited highbury grove school earlier in the day in support of the music in secondary schools trust .", "councillor paul convery , islington council 's executive member for community safety , said : said : ` this is an appalling tragedy , and my deepest sympathies go to this young man 's family and friends .", "` this shocking incident underlines the need for the police , councils , schools and families to keep working together to stop knife crime .", "` knife crime is unacceptable in our community and we support every measure to tackle it . '", "a scotland yard spokesman said : ` the victim was taken by london 's air ambulance to an east london hospital where he died late on wednesday 24 april .", "next of kin are aware . '", "a post-mortem examination will be held in due course .", "an incident room has opened under detective chief inspector chris jones of the homicide and serious crime command . '", "police confirmed a 15-year-old boy has been arrested in connection with the attack and remains in custody at a north london police station .", "one local resident , who did not want to be named , said : ' i have been told that the boy that attacked him went to the back of the bus and stabbed him .", "he then pushed someone over , jumped off the bus and ran through the estate . '", "miriam zoabir , 61 , was picking her 12-year-old granddaughter up from school around 3pm .", "she said : ` there was police and ambulances everywhere . '", "i was so worried my granddaughter might have been caught up in it , but thankfully i saw her quite quickly .", "` there is never really any trouble at this school since my granddaughter started going here . '", "a spokesman for london ambulance said a helicopter ambulance landed in the playing fields of a nearby school before taking the youngster to hospital .", "a woman described hearing a bus screech to a halt moments after the youngster was stabbed on board .", "lucy smith , 56 , who lives in a flat overlooking the scene , said : ' i was looking out of the window . '", "i saw the bus screech to a halt . '", "she said a boy jumped off the bus and ran down the road .", "she said : ` my dogs nearly had him , but i did n't see where he went .", "obviously i did n't know what had happened at the time . '", "paramedics attended the scene along with a helicopter ambulance which landed in the playing fields of nearby highbury grove school in highbury grove .", "one eyewitness , who did not wish to be named , said : ` he was stabbed on the bus , at the back of the bus .", "he looked about 15 years old although some people were saying he might only be 12 . '", "i saw the passengers on the bus milling around and people were giving him cpr .", "i thought at first that he had had some sort of epileptic fit , they were working on him for quite a long time .", "` but then a workman said that it was a boy who had been stabbed .", "` the police and then the ambulance turned up and took over .", "at first there was just one police car and then loads turned up .", "` the next thing was the air ambulance arrived and landed in the car park of highbury grove school .", "` they put the boy on a stretcher with a white screen around him - i do n't think it looked very good . '", "elizabeth johnson , 33 , who lives on the highbury new park estate opposite the scene , described seeing the boy being given cpr as he lay in the road .", "she said : ` they were doing cpr on the road .", "it 's a little child .", "he was very young only year 1 or so .", "` an air ambulance arrived and parked in the school playground and then took off about half an hour later .", "people are still on the bus and they are taking their details . '", "she added that she believed that the young stabbing victim may have been a pupil at a nearby school .", "eyewitness emma nisse said on twitter : ` stabbing on bus right outside my house - police everywhere & paramedics trying to save his life by my garden gate . '", "sorry we are unable to accept comments for legal reasons ."]
# abstract = ["victim was former student at highbury grove school , north london", "teenage attacker ran off through nearby spring gardens estate", "andrew lloyd webber visited the school earlier in the day of attack", "15-year-old arrested in connection with attack and remains in custody"]
file_path = "../dataset/cnndm.test.h5df"
f = h5py.File(file_path,'r')
b = f['dataset'][13]
string = str(b,'utf-8')
data = eval(string)
article = data["article"]
abstract = data["abstract"]
asjdoidaj = data['oracle_sens']

model = SentenceTransformer('roberta-base-nli-stsb-mean-tokens')

sentence_embedding = model.encode(article)
sentenceGraph = createGraph(sentence_embedding)
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine')

reduced_embedding = umap_model.fit_transform(sentence_embedding)
print(reduced_embedding.shape)

# hdbscan = HDBSCAN(min_cluster_size=5, gen_min_span_tree=True)
clu_num = 3

k_means = KMeans(n_clusters=clu_num)
k_means.fit(reduced_embedding)
# hdbscan.fit(reduced_embedding)
y_predict = k_means.predict(reduced_embedding)
# hdb_pre = hdbscan.prediction_data()
count = [0]*clu_num
for i in y_predict:
    count[i]+=1
index_list = np.argpartition(count, -3)[-3:]
centers = k_means.cluster_centers_
# print(centers)
centers_less = [centers[i] for i in index_list]
weights = []
for i in range(len(article)):
    dis = []
    for center in centers_less:
        dis.append(euclidean_distance(np.array(reduced_embedding[i]),np.array(center)))
    weights.append(min(dis)*-1)
newRank = findHighestSimilarityRankTopic(sentenceGraph,[1.0]*len(article),weights)
ans = np.argpartition(newRank, -3)[-3:]

score = get_score(article,abstract,ans)

print()