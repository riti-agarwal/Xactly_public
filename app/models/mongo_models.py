import mongoengine as me  # Use direct import

class Item(me.Document):
    name = me.StringField(required=True)
    image = me.StringField(required=True)
    description = me.StringField(required=True)
    price = me.IntField(required=True)

    # You can add your own fields below

    """
    To get items you can do Item.objects.filter(name="...", ...)
    
    To add a field called embeddings, you will have to add: 

    embeddings = me.ListField(me.FloatField(), blank=True)

    Then, you need to do: 

    i: Item

    i.embeddings = [1, 2, 3, 4.5, ... ]
    i.save()    
    """
