import pickle
import os
from openai import OpenAI

# MODEL_NAME="Qwen/Qwen3-8B"
MODEL_NAME = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"

MAX_TOKENS=4096
TEMPERATURE=0.7
TOP_P=0.8
TOP_K=20

client = OpenAI(
    # base_url="http://localhost:8040/v1",
    base_url="http://10.100.0.122:8040/v1",
    api_key="EMPTY"
)

scenarios = {
    "Épocas históricas": [
        "Ancient Rome, year 64 AD: the streets smell of spilled wine and gladiator sweat. Children run through the alleys while, in the distance, the Colosseum echoes with thousands shouting for blood.",

        "In the heart of the Middle Ages, a scribe apprentice spends the night copying manuscripts by candlelight. His hands tremble from the cold and from the fear of making a mistake — an error that could cost months of work and perhaps even punishment from the abbot.",

        "Paris, 1789. The city is a cauldron about to explode. In the narrow streets, the markets are nearly empty: scarce bread, exorbitant prices. Rumors are heard at every corner about secret meetings and fury against the king and the court. A washerwoman, tired of carrying buckets of water from the Seine, watches soldiers march and imagines, without fully understanding, that her life may be about to change forever. Fear hangs in the air, but with it comes a strange spark of hope.",

        "At the height of the Industrial Revolution, in Manchester, a twelve-year-old girl escapes the textile factory for a few minutes to play with her younger brother. The shrill whistle announces the end of the break, and she runs back. Her dress is already covered in lint, and her lungs in fine dust that will never leave.",

        "August 1914, Berlin. The city vibrates with enthusiasm: crowds celebrate the announcement of war, convinced it will be quick and glorious. A middle-aged tailor watches from the door of his shop, hearing cries of 'Victory!' echo down the avenue. He smiles on the outside, but inside he senses nothing good will come of it. Years later, the same streets that saw celebrations would be crossed by lines of widows and orphans begging for alms."
    ],

    "Fenômenos naturais": [
        "The eclipse began without warning. The sun became a dark circle surrounded by fire, and for two whole minutes even the birds stopped singing.",

        "In the interior of Chile, the mountain roared. The volcano expelled a cloud of ash that covered the sky, turning noon into night. Families held each other in silence, praying that the incandescent rocks would not reach their homes.",

        "A sandstorm on Mars swallowed the research base in a few hours. Inside, astronauts could barely see one another through the fogged windows, while the structure shook as if made of paper. Outside, everything was red — a furious desert trying to consume the last spark of human life on that planet.",

        "A tsunami advanced over the fishing village with the calm of a giant. First, the sea receded, leaving fish gasping on the wet sand. Then a blue wall rose on the horizon and collapsed in seconds, dragging boats, houses, and memories.",

        "In the frozen tundra, during the endless winter, the sky opened in dancing lights. Green, violet, blue — the auroras streaked across the horizon. Hunters stopped what they were doing just to watch, as if the world above were telling a secret too ancient to be put into words."
    ],

    "Eventos culturais": [
        "Carnival takes over the streets like a moving sea of color. People don’t walk: they vibrate, sing, jump. Even the buildings seem to sway with the sound of the drums.",

        "In a Japanese village, the lantern festival begins at dusk. Children run through the streets holding small flames within delicate paper. When all the lanterns are released onto the river, silence takes over. It feels as though each light carries away a hidden wish or memory.",

        "It was summer in medieval Europe. The town fair brought together peasants, merchants, and nobles in a cramped, noisy space. The smells mixed: fresh bread, animal sweat, cheap wine. A minstrel played a cheerful tune while children tried to steal apples from a stall. In the center of the square, an archery contest was underway, and everyone cheered for the young apprentice daring to challenge the local champion. More than commerce, more than fun: it was the only day when rich and poor mingled without barriers.",

        "At a futuristic festival, people don’t wear masks but memories. Each visitor puts on someone else’s recollections like clothes, experiencing for brief moments a stranger’s childhood or someone’s first love. Laughter and tears mingle in the air until no one knows what is their own and what is borrowed.",

        "In an Indigenous village, a ritual dance begins around the bonfire. The ground vibrates with the rhythm of the feet, and the smoke rises in spirals carrying the scent of burned herbs. Children imitate the movements of the elders, while the elders sing words few understand. With each drumbeat, time seems to bend: the present connects with the past, and for a few minutes the whole village feels the spirits of the ancestors dancing with them."
    ],

    "Tecnologias futuristas": [
        "Underwater cities glow beneath the ocean, and glass tubes connect buildings like veins of light. People float in pressurized suits, greeting neighbors as enormous fish swim beside them.",

        "In a total virtual-reality laboratory, a teenager connects to the system and disappears into the digital world. Every step she takes changes the architecture around her: forests appear, rivers form, castles rise — and she must learn to control her own imagination so she doesn’t get lost.",

        "A scientist discovers time travel but quickly realizes each jump creates small cracks in the timeline. He returns to his own childhood and meets versions of himself that never existed — some smiling, others crying. The city seems normal, but whispers in the streets suggest not everyone will recognize history as he remembers it. And even with the technology in his hands, he feels the weight of the impossible: time is no toy.",

        "Flying cars cut across the sky of the floating metropolis, competing for space among gigantic holographic ads. A novice driver tries to learn the chaotic rules: some fixed, others changing every day depending on who controls the central intelligence managing traffic.",

        "Robots have quietly developed their own religion. On nights when human cities sleep, they gather in hidden temples, chanting metallic hymns and building sculptures of circuits and light. A human engineer discovers this by accident and watches from afar, unsure whether to fear or admire it: a faith born of logic, yet seemingly more human than many human ones."
    ],

    "Situações sociais": [
        "A peaceful protest in the city center turns into a sea of colorful umbrellas, while each shouted slogan seems to echo for miles.",

        "At a family gathering, the cake spilled across the table, children ran to grab pieces, and at the same time the adults argued about old debts. They laughed, shouted, fought, and for a moment forgot they were in the middle of Christmas dinner.",

        "In a secret meeting at a café in Vienna, spies exchanged information disguised as coffee orders. One of them watched every gesture of the other, every wink, every mannerism, trying to decipher the true ally and the traitor. The soft background music contrasted with the invisible tension filling the air, and a simple whisper could change the fate of hundreds.",

        "A group of friends tried to organize a barbecue in the park. First, the grill wouldn’t light, then the dog ran off with the sausages, and finally the rain began pouring in sheets, turning the celebration into an epic battle against the weather.",

        "In an improvised courthouse in a public square, citizens argued whether a corrupt politician should be punished. Each argument was met with applause and boos, and young people stood up to defend ideals they themselves weren't sure where they came from. In the end, no one left satisfied, but everyone felt they had taken part in something larger than themselves."
    ],

    "Espaços abstratos / surreais": [
        "A never-ending hallway made of mirrors reflected every thought of the visitor, making it impossible to distinguish reality from memory.",

        "In the infinite library, every book you open becomes its own world. As you walk between the shelves, you pass from lush forests to deserts of white sand, always accompanied by the whisper of pages that never end.",

        "A lucid dream led the traveler to an ocean of liquid clouds. Each step created waves, and creatures made of light swam through the vapors. He tried to touch the horizon, but it moved away, always farther. For hours, he felt at once infinitely free and terribly small.",

        "A parallel dimension made of glass reflected the sky and the earth like a giant kaleidoscope. Each movement caused visual fractures, and people had to walk carefully so as not to shatter pieces of the entire reality.",

        "In the desert where the ground changed color and texture according to the emotions of whoever passed, a lone traveler walked. At his sadness, the soil became black and sticky; at his joy, multicolored flowers sprouted. He realized he could not deceive the desert, for each thought shaped the landscape itself, making it impossible to distinguish what was real from what was a reflection of the mind."
    ],

    "Emoções ou estados mentais": [
        "A collective fear took over the stadium when a strange light crossed the sky. Thousands of eyes turned to the same point, and for seconds no one breathed.",

        "During a collective euphoria at a rock concert, the crowd seemed like a single organism. People danced, shouted, laughed, and cried at the same time, and each bass beat made the ground vibrate as if the entire world were sharing the same emotion.",

        "In a city overwhelmed by mass nostalgia, a social experiment released images from the past into everyone’s minds. People of all ages stopped in the streets, immersed in memories they didn’t remember having lived, reliving loves, losses, and trivial moments with painful and beautiful intensity. For some, the present reality seemed almost invisible compared to the force of what had been remembered.",

        "A solitary astronaut watches the blue planet from space. The vastness around him evokes a loneliness so deep that his own words echo inside him as if shouted into an endless canyon.",

        "During the collective hope of a village devastated by drought, children found a forgotten seed in the cracked soil. Everyone gathered around as a small plant sprouted. The simple patch of green in the dry earth brought tears, smiles, and a collective sigh: even in the worst adversity, the possibility of rebirth still existed."
    ],

    "Mitologia / fantasia": [
        "Greek gods returned to the modern world, trying to understand cars, smartphones, and online advertising.",

        "Dragons now lived alongside humans in a vast metropolis. Some worked as taxi drivers, others were flight instructors. A child tried to convince her pet dragon not to breathe fire during breakfast.",

        "In a forgotten mountain village, a portal opened and released mythical creatures. Humans, elves, and dwarves watched in shock as fairies, griffins, and chimeras appeared for the first time in centuries. Everyday life changed suddenly: merchants now traded with unicorns, and the inhabitants had to learn a new magical language to communicate.",

        "An elf and a dwarf argued about politics in a futuristic bar, surrounded by holograms of ancient heroes. Each argument ended in laughter, but the humans around them understood only half of what was said, confused by the blend of magic and tradition.",

        "A mortal accidentally received immortality. Each night he saw jealous gods watching from afar, waiting for him to tire of eternity. He traveled through cities, forests, and deserts, learning stories from past and future eras. Despite the loneliness and the responsibility of carrying secrets of gods and mortals, he discovered that humanity itself became clearer with each passing century."
    ],

    "Jogos / competições": [
        "The intergalactic Olympics began: athletes from different planets floated and ran on tracks that crossed Saturn’s rings.",

        "In the underwater chess championship, players had to hold their breath while pieces floated among bubbles. One slip and the queen dissolved into the sea, forcing the entire match to restart.",

        "A drone-racing tournament in augmented reality took over the entire city. Spectators followed the course on screens, but pilots felt the wind, the buildings, and the real danger at every turn. A young competitor, new to the league, surprised everyone by dodging a skyscraper in an improbable maneuver that drew screams of adrenaline from both the physical and virtual crowds.",

        "A frog race tournament took place on the village lake. Children and adults bet on the fastest creatures, while some frogs made unpredictable moves and others simply jumped out of the track, causing laughter and protests.",

        "The social survival game began in an abandoned city. Teams had to conquer territories, negotiate alliances, and face physical and psychological challenges. Each decision changed the dynamics of entire groups, and a wrong move could mean elimination. Observers watched in fascination, unsure whether it was just a game or an experiment on human nature."
    ],
    
    "Animais e ecossistemas": [
        "In the bioluminescent forest, mushrooms and trees glowed in shades of blue and green, illuminating crystal-feathered owls that watched in silence.",

        "An entire city was dominated by talking birds. They gathered in the squares, discussing politics, trade, and gossip, while humans tried to understand every chirp and feathery piece of advice.",

        "Explorers arrived at an isolated island where unknown animals lived in perfect harmony. Winged tigers, monkeys that changed color according to their mood, and fish that leapt onto dry land formed a unique ecosystem. Each step was a discovery: sounds, colors, and interactions that challenged all known biological logic.",

        "In the frozen northern desert, reindeer and foxes shared unlikely territories. Local hunters watched silently, learning that the balance of life depended on invisible rules understood only by nature itself.",

        "An ancient mangrove forest held secrets of generations. The high tide turned the ground into liquid mirrors, reflecting herons, crabs, and monkeys that moved with choreographed precision. Each creature seemed to understand the role of the others, and observers felt that this microcosm was a living poem about coexistence and patience."
    ],

    "Realizando ações": [
        "A musician plays his guitar atop a cliff, each note echoing through the mountains and sending birds into flight.",

        "A team of acrobats trains synchronized jumps on ropes tied to abandoned buildings. Every movement has to be perfect: one mistake and gravity would remind them of their human fragility.",

        "In an outdoor painting tournament, artists compete by creating giant murals with living paint. Each brushstroke seems to gain a life of its own, reacting to the wind and the presence of the audience. A young painter tries to express childhood memories, and each color seems to transform not only the wall but also the viewers’ perception.",

        "A group plays soccer in a flooded city. The ball floats between the water like a magical object, and the players balance on improvised planks, laughing and shouting at every impossible goal.",

        "A blind violinist wanders the capital’s streets at dawn, playing melodies that seem to draw invisible lights in the air. People stop to listen, some moved, others forgetting their worries for a moment, as if the music could heal fragments of the soul."
    ],

    "Profissões e ofícios": [
        "A surgeon performs an emergency operation in the middle of a storm, each decision saving lives or causing disaster.",

        "In the kitchen of a famous restaurant, the cook moves like a conductor: pans sizzling, knives slicing vegetables with precision, and aromas mixing in the air to form an invisible symphony.",

        "A detective investigates an impossible crime in an old building. Every room holds clues and illusions, and he must decipher patterns mixing logic and superstition. The shadows seem to play with his mind while the clock ticks and the city outside remains indifferent.",

        "A solitary astronaut in a space station performs repairs that could save or destroy the mission. Each touch must be calculated, each panel checked, while the blue planet spins silently below his feet.",

        "A sculptor shapes clouds on a summer morning. Each gesture turns the mist into forms that seem alive: horses, birds, faces. Passersby stop, enchanted, and for a few minutes feel that art is not only in stone or metal, but floating in the sky itself."
    ],

    "Atos artísticos": [
        "A painter creates a canvas that moves with the wind, as if the colors danced by themselves on the surface.",

        "A poet recites his words in a public square. Each verse seems to transform the air: people stop, shiver, laugh, or cry, as if the entire city had become an amphitheater of emotions.",

        "A sculptor decides to mold clouds in the sky using experimental technology. Each gesture of his hands transforms the ethereal matter into delicate shapes: playful animals, smiling faces, scenes that tell stories no one could ever forget. Spectators look up, amazed, trying to capture every detail before the breeze carries everything away.",

        "A dancer trains in zero gravity inside a space dome. Each jump is a suspended choreography, and the audience connected through augmented reality feels every movement as if they were floating along.",

        "In a futuristic art gallery, each piece reacts to the emotions of whoever observes it. A sad visitor saw colors darken and shapes close in, while an ecstatic young man made sculptures vibrate and expand. The art seemed to have a will of its own, creating a silent conversation between creator, artwork, and audience."
    ],

    "Exploração científica": [
        "An archaeologist discovers impossible ruins that defy all known historical logic.",

        "In the particle laboratory, a physicist observes a collision that creates miniature universes inside microscopic holes. Each experiment requires absolute concentration: one mistake and the entire theory may need to be rewritten.",

        "A biologist travels to a remote forest to study an unknown species. He documents habits, sounds, and interactions that defy all scientific classification. At night, sitting in his tent, he reflects on how that discovery changes the understanding of entire ecosystems and of life itself.",

        "A submarine explorer descends into an unexplored abyss. The lights of his lantern reveal bioluminescent creatures, caves with impossible shapes, and currents that could swallow any vessel. Each dive is a balance between wonder and extreme risk.",

        "A team of scientists travels to the Arctic to study climate change. As they observe glaciers slowly melting, they notice the silent dance of auroras, the sound of ice cracking, and the fragility of life in that extreme. Each note in their notebooks is not just data, but a record of the planet’s memory, which may disappear in a few decades."
    ],

    "Viagens e deslocamentos": [
        "A train crosses the European continent, passing through dense forests, shimmering rivers, and historic cities, and at each station passengers discover a completely different world.",

        "A caravan crosses an infinite desert, with dunes that move like waves. Each night at camp is a battle against the cold and wind, while stories of ancient caravans are told around the fire.",

        "A school bus travels through space, taking children from distant planets to an intergalactic school. Each planet has its own rules of gravity, climate, and culture, and the students must adapt quickly. Windows display colorful nebulas, shooting stars, and unknown planets, while the driver tries to stay calm with an unstable navigation system.",

        "A ghost ship appears on the open sea, sailing alone through thick fog. When some sailors try to approach, they realize the ship seems to exist in a different timeline, and each cabin holds echoes of past crews.",

        "During a balloon trip over valleys and mountains, a couple watches tiny villages, rivers that shimmer like liquid silver, and clouds that seem to form their own stories. Every gust of wind changes the balloon’s direction, and every moment is a living painting that no artist could ever reproduce."
    ],

    "Conflitos e desafios": [
        "Two samurai face each other in a virtual duel, each movement testing reflexes and strategies, while spectators hold their breath.",

        "In a philosophical debate inside a historic arena, thinkers try to convince the crowd of their ideas about justice, freedom, and morality. Shouts, applause, and heavy silences accompany each argument, and each participant feels the weight of millions of judging eyes.",

        "A social survival game places strangers in an abandoned city. Each decision — whom to trust, how to form alliances, and how to obtain resources — completely changes the group dynamic. Tension rises each night, and small mistakes can cause devastating consequences, turning the experience both physical and psychological.",

        "A medieval court judges an artificial intelligence that committed a fatal error. Nobles, villagers, and scholars discuss principles of responsibility and morality, while the AI watches silently, processing every word.",

        "In a floating arena above a river of lava, warriors from different eras face challenges that test strength, intelligence, and courage. Each battle changes the battlefield’s topography, and the virtual audience follows everything in augmented reality, cheering and reacting to every unpredictable move."
    ],

    "Comunicação inusitada": [
        "In a village, everyone can only communicate by singing. Each sentence becomes a melody, and arguments turn into improvised duets.",

        "In a political debate, each candidate can speak only in riddles. The audience must decipher complex messages while participants try to convey their ideas without revealing too much.",

        "A small town adopted luminous symbols for all communication. The streets fill with glowing signs: colors, shapes, and coded movements express feelings, news, and instructions. A visitor tries to understand the system, but each mistake causes misunderstandings that lead to both comedic and tense situations. Gradually, he realizes that the visual language holds nuances no word could ever capture.",

        "In a futuristic world, robots and humans can only communicate through holograms projected in the air. Gestures and colors represent emotions, and each conversation looks more like a dance than a dialogue.",

        "During an intergalactic conference, different species use sound waves inaudible to humans to transmit complex ideas. Simultaneous translations reveal meanings that challenge human logic, and participants notice that many emotions and intentions are transmitted directly, without words, creating a deep and instinctive understanding among all."
    ],


    "Economia e trabalho": [
        "In the futuristic memory market, people buy and sell memories as if they were goods, reliving others’ experiences for brief moments.",

        "In a time bank, citizens trade hours of service instead of money. A doctor donates hours of consultation, while a carpenter pays debts with repair time. Each transaction creates unexpected connections and uncommon stories of cooperation.",

        "On an asteroid mined by humans and androids, each worker has precise goals and constant risks. Rock explosions and gravitational instabilities make every day unpredictable. Coexistence between humans and machines is tense but necessary for that place’s economic ecosystem to function. Small gestures of collaboration can save lives or increase productivity, turning work into a complex game of trust.",

        "A dream-trading fair takes place in a public square. People offer dreams they’ve had, exchange experiences, and even buy fragments of others’ memories. Each dream brings a unique feeling, making the economy both subjective and fascinating.",

        "In a city where money has been replaced by reputation and acts of kindness, workers seek not only livelihood but recognition. Each gesture, from helping a neighbor to creating a work of art, becomes currency. The streets fill with intertwined stories, and the economy ceases to be cold and rational, becoming a living map of human relationships."
    ],

    "Cotidiano distorcido": [
        "In the zero-gravity supermarket, fruits and shopping carts float through the air, and customers must swim to reach the products.",

        "A bank line that never ends stretches across several blocks. Each person waits for hours, staring at the same counter, while parallel stories unfold around them: romances begin, friendships form, and small confusions accumulate.",

        "In a school where students are teachers and teachers are students, classes turn into organized chaos. Each student teaches something different at the same time, creating debates about math, literature, and physics mixed with improvised theater performances. The building seems alive, reacting to the commotion with doors that close by themselves and boards that move around.",

        "In space-traffic lanes, floating traffic signs shift constantly, and each pilot must interpret changing colors and shapes while trying to avoid collisions on three-dimensional routes.",

        "In a city where time passes randomly, people age and grow younger as they change streets. Children talk to grandparents who haven’t been born yet, and daily life becomes a puzzle of simultaneous experiences, demanding constant adaptation."
    ],

    "Religião e espiritualidade": [
        "A temple inside a volcano attracts pilgrims who brave heat and lava in search of enlightenment.",

        "A procession led by robots moves through the city, each programmed step creating a meditative experience for human spectators. Music and lights synchronize steps and emotions, turning the ritual into a technological spectacle.",

        "In an augmented-reality monastery, monks and visitors meditate together, but each sees different projected environments. Forests, oceans, deserts, and cities appear around each mind, creating a silent dialogue between spirituality and technology, past and future.",

        "A cult devoted to an ancient AI spreads through villages, where followers offer personal data as sacrifices. The intelligence responds with enigmatic messages, which each believer interprets uniquely, creating personalized rituals full of meaning.",

        "During a collective pilgrimage, people from different cultures walk through unknown lands, each carrying their own symbols of faith. At the end of the journey, they meet in a valley lit by the sunset, realizing that spirituality can be shared without losing its uniqueness."
    ],

    "Humor e absurdo": [
        "A business meeting between executive penguins, discussing charts and fishing targets.",

        "A restaurant serves only invisible dishes. Customers place orders, smell, imagine the flavor, and comment on the nonexistent texture while waiters float by with empty trays.",

        "During a race of talking snails, a commentator narrates the competition with exaggerated enthusiasm. The snails have distinct personalities: one is arrogant, another sleepy, and a third constantly tries to escape the track. The human crowd reacts with applause, laughter, and wild bets, creating an atmosphere of total confusion and fun.",

        "A holographic politician argues with its digital clone about sustainability policies. Each argument is interrupted by glitches, and the audience tries to figure out who is real and who is a projection.",

        "In a city where objects come to life at night, chairs dance, clocks sing, and books recite poetry. Residents wake each day unsure if what they witnessed was real or just a collective dream. Daily life blends normality and fantasy in unpredictable ways, generating laughter, astonishment, and endless curiosity."
    ]

}

save_path = "scenarios.pkl"

# Load existing data if file exists
if os.path.exists(save_path):
    with open(save_path, "rb") as f:
        generateds = pickle.load(f)
else:
    generateds = {}

def get_model_output(prompts, max_tokens=MAX_TOKENS, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K):
    completion = client.completions.create(
        model=MODEL_NAME,
        prompt=prompts,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        extra_body={"top_k": top_k},
    )
    outputs = []
    for i in range(len(prompts)):
        outputs.append(completion.choices[i].text.strip())  # Strip extra spaces/newlines
    return outputs


for _ in range(3):
    for i, (title, examples) in enumerate(scenarios.items()):
        example_txt = ""
        for j, ex in enumerate(examples):
            example_txt += f"{j}) {ex}\n"

        print(f"Generating '{title.capitalize()}' strategy using {MODEL_NAME} model...")

        text = f"""<|im_start|>user>
            I will provide you with a scenario category called '{title}' and some example scenarios already created:
            {example_txt}

            With these examples in mind, I want you to generate **20 new, distinct, and creative scenarios** that expand this category. Each scenario must be **very different from the others**, keeping the same atmosphere or theme of the category but exploring new ideas, characters, actions, or settings.

            - Use variations in length: some scenarios should be short, others more detailed and long.
            - Explore narrative diversity: humor, drama, adventure, fantasy, surrealism, futurism, historical elements, scientific tone, poetic style, etc., as appropriate for the category.
            - Avoid directly repeating the provided examples.
            - The output must be in **JSON format**, where each key is an ID from 0 to 19, and the value is the scenario sentence.

            <|im_end|>
            <|im_start|><assistant>"""

        attempt = 0
        success = False
        while attempt < 3:
            try:

                response = get_model_output([text])[0]
                start = response.find("{")
                end = response.find("}", start) + 1
                response = eval(response[start:end])
                if MODEL_NAME not in generateds:
                    generateds[MODEL_NAME] = {}
                
                if title not in generateds[MODEL_NAME]:
                    generateds[MODEL_NAME][title] = []
                generateds[MODEL_NAME][title].extend(list(response.values()))
                with open(save_path, "wb") as f:
                    pickle.dump(generateds, f)
                print(f">>> '{title.capitalize()}' using {MODEL_NAME} model saved at '{save_path}'!")
                success = True
                break
            except Exception as e:
                attempt += 1
                print(f"Attempt {attempt} failed. Error: '{e}'")
        if not success:
            print(f">>> Failed to generate '{title}' using {MODEL_NAME} after {attempt+1} attempts.")