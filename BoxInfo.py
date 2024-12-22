class BoxInfo():
    def __init__(self,line):
        words=line.split()
        self.category=words.pop()
        words=[int (string) for string in words]
        self.player_id=words[0]
        del words[0]

        x1,y1,x2,y2,frame_id,lost,grouping,generated=words
        self.box=x1,y1,x2,y2
        self.frame_id=frame_id
        self.lost=lost
        self.grouping=grouping
        self.generated=generated