import weaviate
import weaviate.classes as wvc
from weaviate.classes.config import Property, DataType
import ollama

from tqdm import tqdm
import time

documents = [
    "Tokyo, Latitude 35.6895, Longitude 139.6917",
    "New York, Latitude 40.7128, Longitude -74.0060",
    "Paris, Latitude 48.8566, Longitude 2.3522",
    "London, Latitude 51.5074, Longitude -0.1278",
    "Beijing, Latitude 39.9042, Longitude 116.4074",
    "Moscow, Latitude 55.7558, Longitude 37.6173",
    "Sydney, Latitude -33.8688, Longitude 151.2093",
    "Rio de Janeiro, Latitude -22.9068, Longitude -43.1729",
    "Cape Town, Latitude -33.9249, Longitude 18.4241",
    "Toronto, Latitude 43.6510, Longitude -79.3470",
    "Los Angeles, Latitude 34.0522, Longitude -118.2437",
    "Berlin, Latitude 52.5200, Longitude 13.4050",
    "Mumbai, Latitude 19.0760, Longitude 72.8777",
    "Istanbul, Latitude 41.0082, Longitude 28.9784",
    "Mexico City, Latitude 19.4326, Longitude -99.1332",
    "Bangkok, Latitude 13.7563, Longitude 100.5018",
    "Shanghai, Latitude 31.2304, Longitude 121.4737",
    "Seoul, Latitude 37.5665, Longitude 126.9780",
    "Cairo, Latitude 30.0444, Longitude 31.2357",
    "Buenos Aires, Latitude -34.6037, Longitude -58.3816",
    "Balkan Tensions Rise: Recent skirmishes along the Serbian-Bosnian border have escalated tensions in the Balkans. International observers note an increase in military buildup on both sides. Diplomatic efforts by the EU to mediate the conflict have so far been unsuccessful.",
    "South China Sea Disputes Intensify: The dispute over the Spratly Islands continues to escalate, with multiple nations, including China and the Philippines, increasing their naval presence. Recent satellite images show the construction of new military bases on several islands, raising concerns of a possible confrontation.",
    "Cyberattacks in Eastern Europe: A series of coordinated cyberattacks targeted at government institutions in Ukraine have been traced back to state-sponsored actors in Russia. The attacks disrupted several critical infrastructure systems, leading to heightened cybersecurity alerts across Europe.",
    "Middle East Conflict Widens: The civil war in Syria has expanded, drawing in forces from neighboring countries and increasing the risk of a regional conflict. Recent airstrikes by unidentified planes targeting militia groups have further complicated the peace process.",
    "Renewed Violence in Kashmir: The conflict between India and Pakistan over Kashmir has seen a new wave of violence, with both sides exchanging heavy artillery fire along the Line of Control. The international community calls for restraint and a return to peace talks.",
    "Nuclear Tensions in Korea: North Korea's recent test of a ballistic missile capable of reaching the United States has escalated tensions on the Korean Peninsula. South Korea and Japan have ramped up their defense readiness in response to the growing threat.",
    "Venezuelan Political Crisis: The political crisis in Venezuela worsens as opposition leaders call for mass protests against President Maduro's regime. The economic collapse has driven millions to flee the country, straining regional resources.",
    "Arctic Sovereignty Claims: As global warming opens new passageways in the Arctic, Russia, Canada, and the United States are stepping up their claims in the region. This has led to increased military activity and concerns about potential resource conflicts.",
    "Rohingya Crisis in Myanmar: The situation in Myanmar continues to deteriorate with the Rohingya Muslim minority facing increased violence from government forces. International condemnation has grown, but effective action remains lacking.",
    "peyton farm location 39°06'04.4\"N 104°24'18.0\"W",
    "Moccasin Fire [38.63826,-120.774929]: The 51.1-acre fire in El Dorado County California along Oak Hill Road was 35\% contained as of Thursday, according to Cal Fires Amador-El Dorado unit. The fire, in a rural area the Moccasin Trail west of Somerset, had grown Wednesday afternoon at a moderate rate of spread that was tamped down by firefighters through the night.",
    "Grubbs Fire [39.43511,-121.507399]: The final size of the blaze south of the Thompson Fire and east of Palermo California was 12.8 acres, Cal Fire BTU said in a Thursday morning update. Evacuations were lifted after residents in two zones were told to leave.",
    "Adams Fire [38.91033,-122.607362]: The 16-acre blaze near Lower Lake in Lake County was 60\% contained by Thursday morning after flames broke out on the 16200 block of Main Street. The fire had a “rapid rate” of spread and structures were threatened Wednesday after it broke out just after 3 p.m. All evacuations were lifted, said Cal Fires Sonoma-Lake-Napa unit."
]
rf_data = [
    "Denver international airport uses the following RF frequencies \
    Denver Approach (Final FR-1)	123.850 MHz \
    Denver Approach (Final FR-2)	120.800 MHz \
    Denver Approach (Final FR-3)	125.750 MHz \
    Denver Approach (NE)	124.950 MHz \
    Denver Approach (NW)	119.300 MHz \
    Denver Approach (SE)	126.550 MHz \
    Denver Approach (SW)	120.350 MHz \
    Denver Approach (W)	133.625 MHz \
    Denver Class B (North)	134.850 MHz",
    "Aircraft RF ADS-B is at 1090 MHz"
]

africa_reports = [
    "African Rebel Movements: The Central African Republic is experiencing increased activity by rebel groups, who have taken control of several key towns near 6.6111, 20.9394. The instability has led to a humanitarian crisis, with thousands displaced and in need of aid.",
    "Drought in East Africa: Severe drought conditions in Somalia have worsened near 2.0469, 45.3182, threatening the livelihoods of millions. International aid agencies are calling for urgent food and water supplies to prevent a major humanitarian disaster.",
    "Oil Spill off the Coast of Nigeria: A significant oil spill has been reported near the Niger Delta at 4.9333, 6.3000, causing extensive damage to marine life and affecting the local fishing communities who depend on these waters for their livelihood.",
    "Political Unrest in Zimbabwe: Rising political tensions in Harare at -17.8252, 31.0335 have led to widespread protests and clashes with law enforcement. Observers are concerned about the potential for escalation and the impact on upcoming elections.",
    "Locust Infestation in Kenya: Locust swarms have devastated crops in northern Kenya around 0.0236, 37.9062, posing a serious threat to food security in the region. Efforts to control the outbreak have been hampered by the scale of the infestation and lack of resources."
]

ukraine_reports = [
    "Infrastructure Damage in Kyiv: Recent attacks in the suburbs of Kyiv at 50.4501, 30.5234 have resulted in severe damage to critical infrastructure, including bridges and power stations. Emergency services are working to restore essential services amidst ongoing security challenges.",
    "Civilian Casualties in Donetsk: Heavy shelling in the city of Donetsk at 48.0159, 37.8022 has led to significant civilian casualties and widespread destruction of residential areas. International observers are calling for immediate ceasefire and humanitarian access.",
    "Chemical Plant Explosion in Luhansk: A suspected missile strike caused a massive explosion at a chemical plant near Luhansk at 48.5740, 39.3078. The incident has raised fears of environmental contamination and potential long-term health effects on the local population.",
    "Displacement Crisis in Kharkiv: The intensification of the conflict around Kharkiv at 49.9935, 36.2304 has forced thousands to flee their homes, creating a severe displacement crisis. Aid organizations are struggling to provide shelter and basic necessities to the growing number of refugees.",
    "Port Disruption in Odesa: Military activities near the port of Odesa at 46.4825, 30.7233 have disrupted maritime trade routes, significantly impacting Ukraine's economy. There are reports of several ships damaged and trade operations suspended indefinitely."
]


ukraine_facilities = [
    "Taras Shevchenko National University of Kyiv, Kyiv, Ukraine, 50.4414, 30.5055",
    "Kyiv-Pechersk Lavra Clinic, Kyiv, Ukraine, 50.4347, 30.5572",
    "Zaporizhzhia Nuclear Power Plant, Enerhodar, Ukraine, 47.5114, 34.6507",
    "Ocean Plaza Shopping Mall, Kyiv, Ukraine, 50.4120, 30.5227",
    "Lviv Polytechnic National University, Lviv, Ukraine, 49.8397, 24.0297",
    "Feofaniya Hospital, Kyiv, Ukraine, 50.3480, 30.4629",
    "DniproHES Hydroelectric Power Plant, Zaporizhzhia, Ukraine, 47.8861, 35.0724",
    "Sky Mall, Kyiv, Ukraine, 50.4905, 30.4973",
    "Ivano-Frankivsk National Medical University, Ivano-Frankivsk, Ukraine, 48.9215, 24.7097",
    "Main Military Clinical Hospital, Kyiv, Ukraine, 50.4020, 30.5370",
    "Tsum Kyiv Shopping Center, Kyiv, Ukraine, 50.4483, 30.5224",
    "National University of Kyiv-Mohyla Academy, Kyiv, Ukraine, 50.4658, 30.5188",
    "Lviv National Medical University, Lviv, Ukraine, 49.8407, 24.0305",
    "Dream Town Shopping Mall, Kyiv, Ukraine, 50.4868, 30.4966",
    "Kharkiv National Medical University, Kharkiv, Ukraine, 49.9897, 36.2304",
    "Mechnikov Hospital, Dnipro, Ukraine, 48.4647, 35.0462",
    "Lviv State Aircraft Repair Plant, Lviv, Ukraine, 49.8327, 23.9945",
    "Promenada Center, Ternopil, Ukraine, 49.5515, 25.6056",
    "Zaporizhzhia State Engineering Academy, Zaporizhzhia, Ukraine, 47.8388, 35.1396",
    "Chernihiv Regional Hospital, Chernihiv, Ukraine, 51.4982, 31.2893"
]

ukraine_pois = [
    "Saint Sophia's Cathedral, Kyiv, Ukraine, 50.4529, 30.5143",
    "Lviv Historical Center, Lviv, Ukraine, 49.8419, 24.0315",
    "Odesa Opera and Ballet Theater, Odesa, Ukraine, 46.4856, 30.7430",
    "Kamianets-Podilskyi Castle, Kamianets-Podilskyi, Ukraine, 48.6750, 26.5809",
    "Chernobyl Exclusion Zone, Chernobyl, Ukraine, 51.2707, 30.2219",
    "Lavra Monastery, Kyiv, Ukraine, 50.4345, 30.5592",
    "Pechersk Lavra, Kyiv, Ukraine, 50.4345, 30.5592",
    "The Motherland Monument, Kyiv, Ukraine, 50.4267, 30.5638",
    "Rynok Square, Lviv, Ukraine, 49.8419, 24.0315",
    "Carpathian National Nature Park, Carpathians, Ukraine, 48.2629, 24.3500",
    "Swallow's Nest Castle, Crimea, Ukraine, 44.4331, 34.1224",
    "Potemkin Stairs, Odesa, Ukraine, 46.4873, 30.7438",
    "Askania-Nova Biosphere Reserve, Askania-Nova, Ukraine, 46.4599, 33.8736",
    "Zaporizhzhia Sich, Khortytsia, Ukraine, 47.8388, 35.1396",
    "Vorontsov Palace, Alupka, Crimea, Ukraine, 44.4184, 34.0456",
    "The Last Barricade Museum, Kyiv, Ukraine, 50.4501, 30.5241",
    "Livadia Palace, Yalta, Crimea, Ukraine, 44.4695, 34.1429",
    "Museum of Folk Architecture and Rural Life, Lviv, Ukraine, 49.8075, 24.0986",
    "Massandra Palace, Yalta, Crimea, Ukraine, 44.5191, 34.2008",
    "Derzhprom Building, Kharkiv, Ukraine, 49.9935, 36.2304"
]

points_of_interest = [
    "Eiffel Tower, France, 48.8584, 2.2945",
    "Great Wall of China, China, 40.4319, 116.5704",
    "Statue of Liberty, USA, 40.6892, -74.0445",
    "Machu Picchu, Peru, -13.1631, -72.5450",
    "Colosseum, Italy, 41.8902, 12.4922",
    "Pyramids of Giza, Egypt, 29.9792, 31.1342",
    "Taj Mahal, India, 27.1751, 78.0421",
    "Grand Canyon, USA, 36.1069, -112.1129",
    "Christ the Redeemer, Brazil, -22.9519, -43.2105",
    "Acropolis of Athens, Greece, 37.9715, 23.7257",
    "Sydney Opera House, Australia, -33.8568, 151.2153",
    "Mount Fuji, Japan, 35.3606, 138.7274",
    "Stonehenge, UK, 51.1789, -1.8262",
    "Sagrada Familia, Spain, 41.4036, 2.1744",
    "Niagara Falls, Canada/USA, 43.0962, -79.0377",
    "Petra, Jordan, 30.3285, 35.4444",
    "Angkor Wat, Cambodia, 13.4125, 103.8667",
    "Victoria Falls, Zambia/Zimbabwe, -17.9243, 25.8572",
    "Louvre Museum, France, 48.8606, 2.3376",
    "Banff National Park, Canada, 51.4968, -115.9281",
    "Galapagos Islands, Ecuador, -0.9538, -90.9656",
    "Table Mountain, South Africa, -33.9628, 18.4098",
    "Yellowstone National Park, USA, 44.4237, -110.5885",
    "Santorini, Greece, 36.3932, 25.4615",
    "Chichen Itza, Mexico, 20.6843, -88.5678",
    "Neuschwanstein Castle, Germany, 47.5576, 10.7498",
    "Potala Palace, Tibet, 29.6578, 91.1169",
    "Buckingham Palace, UK, 51.5014, -0.1419",
    "Mount Everest, Nepal/Tibet, 27.9881, 86.9250",
    "Yosemite National Park, USA, 37.8651, -119.5383",
    "Bora Bora, French Polynesia, -16.5004, -151.7415",
    "Golden Gate Bridge, USA, 37.8199, -122.4783",
    "Mount Kilimanjaro, Tanzania, -3.0674, 37.3556",
    "Las Vegas Strip, USA, 36.1147, -115.1728",
    "Palace of Versailles, France, 48.8049, 2.1204",
    "Alhambra, Spain, 37.1761, -3.5881",
    "Ayers Rock (Uluru), Australia, -25.3444, 131.0369",
    "Forbidden City, China, 39.9163, 116.3975",
    "Mecca, Saudi Arabia, 21.4225, 39.8262",
    "Hollywood Sign, USA, 34.1341, -118.3217",
    "Kremlin, Russia, 55.7517, 37.6176",
    "Capitol Hill, USA, 38.8899, -77.0091",
    "Disney World, USA, 28.3852, -81.5639",
    "Eden Project, UK, 50.3601, -4.7445",
    "Iguazu Falls, Argentina/Brazil, -25.6953, -54.4367",
    "Sphinx of Giza, Egypt, 29.9753, 31.1376",
    "Piazza San Marco, Italy, 45.4338, 12.3378",
    "Prague Castle, Czech Republic, 50.0911, 14.4012",
    "Red Square, Russia, 55.7539, 37.6208",
    "Patagonia, Argentina/Chile, -49.3315, -72.8863",
    "Dubrovnik Old Town, Croatia, 42.6419, 18.1083",
    "Blue Mosque, Turkey, 41.0053, 28.9768",
    "Pamukkale, Turkey, 37.9184, 29.1257",
    "Everglades, USA, 25.2866, -80.8987",
    "Great Barrier Reef, Australia, -18.2871, 147.6992",
    "Kakadu National Park, Australia, -12.8035, 132.4789",
    "Death Valley, USA, 36.5054, -117.0794",
    "Salar de Uyuni, Bolivia, -20.1338, -67.4891",
    "Oxford University, UK, 51.7566, -1.2547",
    "Cinque Terre, Italy, 44.1281, 9.7124",
    "Lake Baikal, Russia, 53.5587, 108.1650",
    "Hagia Sophia, Turkey, 41.0086, 28.9802",
    "Bora Bora, French Polynesia, -16.5004, -151.7415",
    "Fjords of Norway, Norway, 60.4720, 5.0760",
    "Matterhorn, Switzerland, 45.9763, 7.6586",
    "Plitvice Lakes, Croatia, 44.8654, 15.5820",
    "Timbuktu, Mali, 16.7666, -3.0026",
    "Uluru, Australia, -25.3444, 131.0369",
    "Abu Simbel, Egypt, 22.3372, 31.6253",
    "Kruger National Park, South Africa, -23.9884, 31.5547",
    "Teotihuacan, Mexico, 19.6928, -98.8438",
    "Berlin Wall, Germany, 52.5163, 13.3777",
    "Anne Frank House, Netherlands, 52.3752, 4.8840",
    "Mount Olympus, Greece, 40.0856, 22.3603",
    "Angkor Thom, Cambodia, 13.4419, 103.8586",
    "Terra Cotta Warriors, China, 34.3833, 109.2770",
    "Mont Saint-Michel, France, 48.6360, -1.5115",
    "Auschwitz, Poland, 50.0369, 19.1754",
    "Moai Statues, Easter Island, -27.1127, -109.3497",
    "Great Ocean Road, Australia, -38.6805, 143.3916",
    "Maldives Islands, Maldives, 3.2028, 73.2207",
    "Banff, Canada, 51.1784, -115.5708",
    "Yucatan Peninsula, Mexico, 20.7099, -89.0943",
    "Zanzibar, Tanzania, -6.1659, 39.2026",
    "Isle of Skye, UK, 57.5359, -6.2263",
    "Grand Bazaar, Turkey, 41.0107, 28.9684",
    "Piazza del Duomo, Italy, 43.7732, 11.2556",
    "Hermitage Museum, Russia, 59.9398, 30.3146",
    "Alcatraz Island, USA, 37.8270, -122.4230",
    "Blue Lagoon, Iceland, 63.8804, -22.4495"
]

sea_ports = [
    "Port of Shanghai, China, 31.3666, 121.6033",
    "Port of Shenzhen, China, 22.5, 113.883333",
    "Port of Singapore, Singapore, 1.2647, 103.8222",
    "Port of Rotterdam, Netherlands, 51.9561, 4.1358",
    "Port of Ningbo-Zhoushan, China, 29.9481, 121.8841",
    "Port of Guangzhou, China, 23.0971, 113.319",
    "Port of Busan, South Korea, 35.1018, 129.0304",
    "Port of Hong Kong, Hong Kong, 22.3193, 114.1694",
    "Port of Qingdao, China, 36.0648, 120.3747",
    "Port of Dubai, UAE, 25.2629, 55.3284",
    "Port of Tianjin, China, 39.1256, 117.7068",
    "Port of Antwerp, Belgium, 51.2276, 4.4054",
    "Port of Kaohsiung, Taiwan, 22.5771, 120.3128",
    "Port of Hamburg, Germany, 53.5466, 9.9849",
    "Port of Felixstowe, UK, 51.9638, 1.3512",
    "Port of Los Angeles, USA, 33.7365, -118.2923",
    "Port of Long Beach, USA, 33.7542, -118.2165",
    "Port of New York and New Jersey, USA, 40.6699, -74.1782",
    "Port of Santos, Brazil, -23.9630, -46.3365",
    "Port of Colombo, Sri Lanka, 6.9359, 79.8483",
    "Port of Valencia, Spain, 39.4440, -0.3193",
    "Port of Laem Chabang, Thailand, 13.1007, 100.8835",
    "Port of Basuo, China, 19.099762, 108.620575",
    "Port of Manila, Philippines, 14.5895, 120.9839",
    "Port of Bremerhaven, Germany, 53.5522, 8.5865",
    "Port of Durban, South Africa, -29.8692, 31.0455",
    "Port of Sydney, Australia, -33.8548, 151.2165",
    "Port of Melbourne, Australia, -37.8140, 144.9633",
    "Port of Jeddah, Saudi Arabia, 21.4868, 39.1862",
    "Port of Jakarta, Indonesia, -6.1165, 106.8650",
    "Port of Vancouver, Canada, 49.2827, -123.1207",
    "Port of Richards Bay, South Africa, -28.7807, 32.0383",
    "Port of Salalah, Oman, 16.9344, 54.0924",
    "Port of Nantong, China, 32.751389, 120.815"
    "Port of Rotterdam, Netherlands, 51.9561, 4.1358",
    "Port of Dalian, China, 38.9215, 121.6392",
    "Port of Xiamen, China, 24.502262, 118.084531",
    "Port of Yingkou, China, 40.278889, 122.095",
    "Port of Chennai, India, 13.0827, 80.2707",
    "Port of Rizhao, China, 35.383333, 119.558333",
    "Port of Karachi, Pakistan, 24.8465, 67.0248",
    "Port of Marseille, France, 43.2965, 5.3698",
    "Port of Gothenburg, Sweden, 57.7089, 11.9746",
    "Port of Algeciras, Spain, 36.1408, -5.4562",
    "Port of Mumbai, India, 19.0760, 72.8777",
    "Port of Istanbul, Turkey, 41.0082, 28.9784",
    "Port of San Francisco, USA, 37.7749, -122.4194",
    "Port of Seattle, USA, 47.6062, -122.3321",
    "Port of Miami, USA, 25.7617, -80.1918",
    "Port of Alexandria, Egypt, 31.2001, 29.9187",
    "Port of Montreal, Canada, 45.5017, -73.5673",
    "Port of Copenhagen, Denmark, 55.6761, 12.5683",
    "Port of Oslo, Norway, 59.9139, 10.7522",
    "Port of Stockholm, Sweden, 59.3293, 18.0686",
    "Port of Piraeus, Greece, 37.9357, 23.6471",
    "Port of Helsinki, Finland, 60.1695, 24.9354",
    "Port of Buenos Aires, Argentina, -34.6037, -58.3816",
    "Port of Veracruz, Mexico, 19.1738, -96.1342",
    "Port of Lima, Peru, -12.0464, -77.0428",
    "Port of Cape Town, South Africa, -33.9249, 18.4241",
    "Port of Reykjavik, Iceland, 64.1466, -21.9426",
    "Port of Tokyo, Japan, 35.615652, 139.796908", 
    "Port of Yokohama, Japan, 35.4437, 139.6380",
    "Port of Mombasa, Kenya, -4.0500, 39.6668",
    "Port of Lagos, Nigeria, 6.5244, 3.3792",
    "Port of Casablanca, Morocco, 33.5731, -7.5898",
    "Port of Rio de Janeiro, Brazil, -22.9068, -43.1729",
    "Port of Helsinki, Finland, 60.1695, 24.9354",
    "Port of Auckland, New Zealand, -36.8485, 174.7633",
    "Port of Wellington, New Zealand, -41.2865, 174.7762",
    "Port of Tallinn, Estonia, 59.4370, 24.7536",
    "Port of Riga, Latvia, 56.9496, 24.1052",
    "Port of Odessa, Ukraine, 46.4825, 30.7233",
    "Port of Helsinki, Finland, 60.1695, 24.9354",
    "Port of Antwerp, Belgium, 51.2276, 4.4054",
    "Port of Le Havre, France, 49.4944, 0.1079",
    "Port of Genoa, Italy, 44.4056, 8.9463",
    "Port of Izmir, Turkey, 38.4237, 27.1428",
    "Port of Barcelona, Spain, 41.3851, 2.1734",
    "Port of Marseille, France, 43.2965, 5.3698"
]

additional_airports = [
    "Melbourne Airport, -37.6690, 144.8410",
    "Leonardo da Vinci–Fiumicino Airport, Rome, 41.7999, 12.2462",
    "Taiwan Taoyuan International Airport, 25.0777, 121.2328",
    "Malaga Airport, 36.6749, -4.4991",
    "Manchester Airport, UK, 53.3651, -2.2728",
    "Munich Airport, 48.3538, 11.7861",
    "Charles de Gaulle Airport, Paris, 49.0097, 2.5479",
    "Narita International Airport, Tokyo, 35.7720, 140.3929",
    "Jorge Chavez International Airport, Lima, -12.0219, -77.1143",
    "Dublin Airport, 53.4213, -6.2701",
    "Brisbane Airport, -27.3842, 153.1176",
    "Helsinki Airport, 60.3172, 24.9633",
    "Edinburgh Airport, 55.9500, -3.3725",
    "Antalya Airport, 36.8987, 30.8005",
    "Tan Son Nhat International Airport, Ho Chi Minh City, 10.8185, 106.6518",
    "Zurich Airport, 47.4582, 8.5481",
    "Sao Paulo Congonhas Airport, -23.6261, -46.6564",
    "Salt Lake City International Airport, 40.7899, -111.9791",
    "Copenhagen Airport, 55.6180, 12.6560",
    "Budapest Airport, 47.4369, 19.2556",
    "Hamad International Airport, Doha, 25.2731, 51.6081",
    "Stockholm Arlanda Airport, 59.6519, 17.9186",
    "Perth Airport, -31.9403, 115.9672",
    "Osaka Itami Airport, 34.7855, 135.4380",
    "Bogota El Dorado International Airport, 4.7017, -74.1469",
    "King Khalid International Airport, Riyadh, 24.9576, 46.6988",
    "King Khaled International Airport, Riyadh, 24.9576, 46.6988",
    "Adelaide Airport, -34.9450, 138.5306",
    "Porto Airport, 41.2481, -8.6814",
    "Nice Cote d'Azur Airport, 43.6584, 7.2159",
    "Jeddah King Abdulaziz International Airport, 21.6796, 39.1565",
    "Birmingham Airport, UK, 52.4539, -1.7480",
    "Nairobi Jomo Kenyatta International Airport, -1.3192, 36.9278",
    "Glasgow International Airport, 55.8719, -4.4330",
    "Detroit Metropolitan Airport, 42.2162, -83.3554",
    "Portland International Airport, Oregon, 45.5898, -122.5975",
    "St. Petersburg Pulkovo Airport, 59.8003, 30.2625",
    "Lyon Saint Exupery Airport, 45.7256, 5.0811",
    "Wellington Airport, -41.3272, 174.8053",
    "Queen Alia International Airport, Amman, 31.7226, 35.9932",
    "Tampa International Airport, 27.9755, -82.5332",
    "Vancouver International Airport, 49.1947, -123.1834",
    "Luxembourg Airport, 49.6266, 6.2115",
    "Philadelphia International Airport, 39.8744, -75.2424",
    "Kuala Lumpur International Airport, 2.7456, 101.7072",
    "Milan Malpensa Airport, 45.6301, 8.7281",
    "Jakarta Soekarno-Hatta International Airport, -6.1275, 106.6537",
    "Prague Vaclav Havel Airport, 50.1008, 14.26",
    "Kingston Norman Manley International Airport, 17.9357, -76.7875"
]

airports = [
    "Hartsfield-Jackson Atlanta International Airport, 33.6407, -84.4277",
    "Beijing Capital International Airport, 40.0801, 116.5846",
    "Los Angeles International Airport, 33.9416, -118.4085",
    "Tokyo Haneda Airport, 35.5494, 139.7798",
    "Dubai International Airport, 25.2532, 55.3657",
    "Paris Charles de Gaulle Airport, 49.0097, 2.5479",
    "Heathrow Airport, London, 51.4700, -0.4543",
    "Shanghai Pudong International Airport, 31.1443, 121.8083",
    "Chicago O'Hare International Airport, 41.9742, -87.9073",
    "Hong Kong International Airport, 22.3080, 113.9185",
    "Amsterdam Schiphol Airport, 52.3105, 4.7683",
    "Madrid Barajas Airport, 40.4983, -3.5676",
    "John F. Kennedy International Airport, New York, 40.6413, -73.7781",
    "Singapore Changi Airport, 1.3644, 103.9915",
    "Guangzhou Baiyun International Airport, 23.3959, 113.3080",
    "Frankfurt Airport, 50.0379, 8.5622",
    "Incheon International Airport, 37.4602, 126.4407",
    "Soekarno-Hatta International Airport, Jakarta, -6.1275, 106.6537",
    "Dallas/Fort Worth International Airport, 32.8998, -97.0403",
    "Istanbul Airport, 41.2753, 28.7519",
    "Suvarnabhumi Airport, Bangkok, 13.6900, 100.7501",
    "Kuala Lumpur International Airport, 2.7456, 101.7072",
    "San Francisco International Airport, 37.6213, -122.3790",
    "Denver International Airport, 39.8561, -104.6737",
    "Barcelona El Prat Airport, 41.2974, 2.0833",
    "Chhatrapati Shivaji Maharaj International Airport, Mumbai, 19.0896, 72.8656",
    "Toronto Pearson International Airport, 43.6777, -79.6248",
    "Sydney Kingsford Smith Airport, -33.9399, 151.1753",
    "Miami International Airport, 25.7959, -80.2871",
    "São Paulo Guarulhos International Airport, -23.4356, -46.4731",
    "Orlando International Airport, 28.4312, -81.3081",
    "Moscow Sheremetyevo Airport, 55.9726, 37.4146",
    "Cairo International Airport, 30.1219, 31.4056",
    "Kansai International Airport, Osaka, 34.4352, 135.2440",
    "Seattle-Tacoma International Airport, 47.4502, -122.3088",
    "Chengdu Shuangliu International Airport, 30.5785, 103.9471",
    "Ben Gurion Airport, Tel Aviv, 32.0055, 34.8854",
    "Newark Liberty International Airport, 40.6895, -74.1745",
    "Brussels Airport, 50.9014, 4.4844",
    "Santiago International Airport, -33.3930, -70.7858",
    "Narita International Airport, Tokyo, 35.7720, 140.3929",
    "George Bush Intercontinental Airport, Houston, 29.9902, -95.3368",
    "Lisbon Airport, 38.7742, -9.1342",
    "Mexico City International Airport, 19.4363, -99.0721",
    "Cape Town International Airport, -33.9715, 18.6021",
    "Schiphol Airport, Amsterdam, 52.3105, 4.7683",
    "Vienna International Airport, 48.1103, 16.5697",
    "Athens International Airport, 37.9364, 23.9475",
    "Oslo Airport, 60.1976, 11.1004"
]

documents = documents + airports + additional_airports + sea_ports + points_of_interest + ukraine_facilities + ukraine_pois + ukraine_reports + africa_reports + rf_data


while True:
    try:
        client = weaviate.connect_to_local()
        client.is_live()  # Check if Weaviate is live
        print("Connected to Weaviate successfully.")
        break
    except Exception as e:
        print("Waiting for Weaviate to be available...")
        time.sleep(5)  # Wait before retrying

#Only do this once
client = weaviate.connect_to_local()
# Create a new data collection

collection_name = "docs"

try:
    client.collections.delete(collection_name)
except:
    print("Delete didnt work (probably not an issue)")

collection = client.collections.create(
    name = collection_name, # Name of the data collection
    properties=[
        Property(name="text", data_type=DataType.TEXT), # Name and data type of the property
    ],
)

with collection.batch.dynamic() as batch:
  for i, d in tqdm(enumerate(documents), total=len(documents)):
    response = ollama.embeddings(model = "all-minilm",
                                 prompt = d)
    # Add data object with text and embedding
    batch.add_object(
        properties = {"text" : d},
        vector = response["embedding"],
    )

client.close()

print("done!")
